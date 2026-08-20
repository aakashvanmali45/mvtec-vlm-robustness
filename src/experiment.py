"""Experiment runners: orchestrate model sweeps over categories and strategies."""

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data import collect_test_samples, load_prompts
from src.metrics import compute_metrics
from src.models import load_classifier, zero_shot_classify_category

from src.training import (
    LoRATrainingConfig,
    attach_lora,
    sample_few_shot_split,
    set_all_seeds,
    train_lora_adapter,
)


def run_zero_shot_sweep(
    prompts_config: str | Path,
    data_root: str | Path,
    models: list[str],
    strategies: list[str],
    categories: list[str] | str,
    output_path: str | Path,
    batch_size: int = 16,
    device: str = "cuda",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run zero-shot classification across models x categories x strategies.

    Args:
        prompts_config: Path to YAML prompt configuration.
        data_root: Path to MVTec-AD dataset root.
        models: List of model names to sweep (must be in MODEL_REGISTRY).
        strategies: List of prompt strategies to sweep.
        categories: Either 'all' (use every category in prompts_config) or a list.
        output_path: Where to save the results CSV.
        batch_size: Inference batch size.
        device: 'cuda' or 'cpu'.
        verbose: If True, print per-configuration progress.

    Returns:
        DataFrame with one row per (model, category, strategy) with all metrics.
    """
    prompts_all = load_prompts(prompts_config)

    if categories == "all":
        categories = sorted(prompts_all.keys())
    elif isinstance(categories, str):
        raise ValueError(f"categories must be 'all' or a list, got string {categories!r}")

    # Validate up front so we fail fast instead of after loading a model.
    for cat in categories:
        if cat not in prompts_all:
            raise KeyError(f"Category '{cat}' not in prompts config")
        for strat in strategies:
            if strat not in prompts_all[cat]:
                raise KeyError(f"Strategy '{strat}' missing for category '{cat}'")

    rows = []
    t_start = time.time()

    for model_name in models:
        if verbose:
            print(f"\n[loading {model_name}]")
        classifier = load_classifier(model_name, device=device)

        for category in categories:
            samples = collect_test_samples(category, data_root)

            for strategy in strategies:
                prompts = prompts_all[category][strategy]
                try:
                    results = zero_shot_classify_category(
                        classifier, samples, prompts, batch_size=batch_size
                    )
                    metrics = compute_metrics(results)
                    row = {
                        "model": model_name,
                        "category": category,
                        "strategy": strategy,
                        "n_total": metrics["n_total"],
                        "n_good": metrics["n_good"],
                        "n_defective": metrics["n_defective"],
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "auroc": metrics["auroc"],
                        "confusion_matrix": str(metrics["confusion_matrix"]),
                        "prompt_good": prompts[0],
                        "prompt_defective": prompts[1],
                    }
                    rows.append(row)

                    if verbose:
                        elapsed = time.time() - t_start
                        print(
                            f"[{elapsed:6.1f}s] {model_name:>6} | {category:<12} | "
                            f"{strategy:<18} | bal_acc={metrics['balanced_accuracy']:.3f}  "
                            f"AUROC={metrics['auroc']:.3f}"
                        )
                except Exception as e:
                    print(f"[ERROR] {model_name} | {category} | {strategy}: {e}")

        # Free the model from GPU memory before loading the next one.
        del classifier

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nDone. {len(df)} rows saved to {output_path}")
        print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")

    return df

def run_few_shot_sweep(
    prompts_config: str | Path,
    data_root: str | Path,
    models: list[str],
    categories: list[str] | str,
    k_values: list[int],
    seeds: list[int],
    strategy: str,
    output_path: str | Path,
    lora_config: dict[str, Any] | None = None,
    device: str = "cuda",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run few-shot LoRA fine-tuning sweep across models x categories x k x seeds.

    For each combination:
      1. Load a fresh classifier and attach fresh LoRA adapters.
      2. Sample a k-shot train/eval split (seeded).
      3. Train the adapter on the k+k training images.
      4. Evaluate on the held-out split.
      5. Record metrics and training loss trajectory.

    Args:
        prompts_config: Path to YAML prompt configuration.
        data_root: Path to MVTec-AD dataset root.
        models: List of model names (currently only 'clip' is supported for LoRA).
        categories: Either 'all' or an explicit list.
        k_values: List of shots per class (e.g. [5] for k=5 only).
        seeds: List of seeds; each (model, category, k) is trained once per seed.
        strategy: Prompt strategy key (e.g. 'naive').
        output_path: Where to save the results CSV.
        lora_config: Optional dict of LoRA hyperparameter overrides.
            Any key from LoRATrainingConfig can be overridden.
        device: Torch device.
        verbose: Per-configuration progress prints.

    Returns:
        DataFrame with one row per (model, category, k, seed) with metrics + loss trajectory.
    """
    prompts_all = load_prompts(prompts_config)

    if categories == "all":
        categories = sorted(prompts_all.keys())
    elif isinstance(categories, str):
        raise ValueError(f"categories must be 'all' or a list, got string {categories!r}")

    # Validate categories and strategy up front.
    for cat in categories:
        if cat not in prompts_all:
            raise KeyError(f"Category '{cat}' not in prompts config")
        if strategy not in prompts_all[cat]:
            raise KeyError(f"Strategy '{strategy}' missing for category '{cat}'")

    # Build LoRA training config with any overrides from lora_config dict.
    training_config = LoRATrainingConfig(**(lora_config or {}))

    # Only CLIP is currently supported for LoRA (SigLIP peft integration is less mature).
    supported_models = {"clip", "siglip"}
    for m in models:
        if m not in supported_models:
            raise NotImplementedError(
                f"LoRA fine-tuning only supported for 'clip' and 'siglip' currently, got '{m}'"
            )

    rows = []
    t_start = time.time()
    n_configs = len(models) * len(categories) * len(k_values) * len(seeds)
    config_idx = 0

    for model_name in models:
        for category in categories:
            samples = collect_test_samples(category, data_root)
            prompts = prompts_all[category][strategy]

            for k in k_values:
                for seed in seeds:
                    config_idx += 1

                    try:
                        # Reproducibility: seed everything before both split and training.
                        set_all_seeds(seed)

                        # Split with the same seed.
                        train_samples, eval_samples = sample_few_shot_split(
                            samples, k=k, seed=seed
                        )

                        # Fresh model and fresh adapters for every run.
                        classifier = load_classifier(model_name, device=device)
                        attach_lora(classifier.model, training_config)

                        # Train.
                        if verbose:
                            print(
                                f"\n[{config_idx}/{n_configs}] {model_name} | {category} | "
                                f"k={k} | seed={seed} | training..."
                            )
                        epoch_losses = train_lora_adapter(
                            classifier=classifier,
                            train_samples=train_samples,
                            prompts=prompts,
                            config=training_config,
                            device=device,
                            verbose=False,  # too noisy inside the sweep
                        )

                        # Evaluate.
                        eval_results = zero_shot_classify_category(
                            classifier, eval_samples, prompts,
                            batch_size=training_config.batch_size * 2,
                        )
                        metrics = compute_metrics(eval_results)

                        row = {
                            "model": model_name,
                            "category": category,
                            "k": k,
                            "seed": seed,
                            "strategy": strategy,
                            "n_train": len(train_samples),
                            "n_eval": len(eval_samples),
                            "n_eval_good": metrics["n_good"],
                            "n_eval_defective": metrics["n_defective"],
                            "accuracy": metrics["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1": metrics["f1"],
                            "auroc": metrics["auroc"],
                            "confusion_matrix": str(metrics["confusion_matrix"]),
                            "final_train_loss": epoch_losses[-1],
                            "first_train_loss": epoch_losses[0],
                            "epoch_losses": str(epoch_losses),
                            "lora_rank": training_config.rank,
                            "lora_alpha": training_config.alpha,
                            "lora_target_modules": ",".join(training_config.target_modules),
                            "lora_lr": training_config.learning_rate,
                            "lora_epochs": training_config.epochs,
                        }
                        rows.append(row)

                        if verbose:
                            elapsed = time.time() - t_start
                            print(
                                f"  [{elapsed:6.1f}s] bal_acc={metrics['balanced_accuracy']:.3f}  "
                                f"AUROC={metrics['auroc']:.3f}  "
                                f"loss {epoch_losses[0]:.3f}→{epoch_losses[-1]:.3f}"
                            )

                    except Exception as e:
                        print(
                            f"[ERROR] {model_name} | {category} | k={k} | seed={seed}: {e}"
                        )
                    finally:
                        # Free GPU memory before the next run.
                        if "classifier" in locals():
                            del classifier
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nDone. {len(df)} rows saved to {output_path}")
        print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")

    return df