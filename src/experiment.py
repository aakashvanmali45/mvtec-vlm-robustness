"""Experiment runners: orchestrate model sweeps over categories and strategies."""

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import numpy as np

from src.data import collect_test_samples, load_prompts
from src.metrics import compute_metrics
from src.models import load_classifier, zero_shot_classify_category

from src.calibration import find_optimal_threshold, apply_threshold_to_results
from src.data import sample_three_way_split
from src.training import load_adapter, save_adapter

from src.training import (
    LoRATrainingConfig,
    attach_lora,
    sample_few_shot_split,
    set_all_seeds,
    train_lora_adapter,
)

from PIL import Image
from src.corruptions import CORRUPTION_TYPES, SEVERITY_LEVELS, corrupt_image

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
    adapter_save_dir: str | Path | None = None,
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

                        if adapter_save_dir is not None:
                            adapter_path = Path(adapter_save_dir) / f"{model_name}_{category}_k{k}_seed{seed}"

                        else:
                            adapter_path = None
                        epoch_losses = train_lora_adapter(
                            classifier=classifier,
                            train_samples=train_samples,
                            prompts=prompts,
                            config=training_config,
                            device=device,
                            verbose=False,  # too noisy inside the sweep
                            save_adapter_path=adapter_path,
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

def run_corruption_evaluation(
    prompts_config: str | Path,
    data_root: str | Path,
    models: list[str],
    categories: list[str] | str,
    strategy: str,
    fewshot_adapter_dir: str | Path | None,
    output_path: str | Path,
    corruption_types: list[str] | None = None,
    severity_levels: list[int] | None = None,
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate model robustness under image corruptions.

    For each (model, category, corruption_type, severity) combination,
    applies the corruption to test images and computes classification metrics.
    This is inference-only — no retraining. Training remains on clean images.

    Args:
        prompts_config: Path to YAML prompts.
        data_root: Path to MVTec-AD.
        models: List of model names.
        categories: 'all' or list.
        strategy: Prompt strategy (e.g. 'naive').
        fewshot_adapter_dir: If set, loads few-shot LoRA adapters for each
            (model, category) from this directory. Adapter path convention:
            {dir}/{model}_{category}_k5_seed{seed}. Currently unused —
            corruption evaluation runs zero-shot only in this initial version.
            Included in signature for future few-shot corruption evaluation.
        output_path: Where to save the CSV.
        corruption_types: Which corruptions to run. Defaults to all 5.
        severity_levels: Which severities. Defaults to all 3.
        batch_size: Inference batch size.
        device: Torch device.
        seed: Random seed for noise-based corruptions.
        verbose: Print per-configuration progress.

    Returns:
        DataFrame with one row per (model, category, corruption, severity).
    """
    prompts_all = load_prompts(prompts_config)

    if categories == "all":
        categories = sorted(prompts_all.keys())
    elif isinstance(categories, str):
        raise ValueError(f"categories must be 'all' or a list, got string {categories!r}")

    corruption_types = corruption_types or list(CORRUPTION_TYPES)
    severity_levels = severity_levels or list(SEVERITY_LEVELS)

    for c in corruption_types:
        if c not in CORRUPTION_TYPES:
            raise ValueError(f"Unknown corruption '{c}'. Valid: {list(CORRUPTION_TYPES)}")
    for s in severity_levels:
        if s not in SEVERITY_LEVELS:
            raise ValueError(f"Invalid severity {s}. Valid: {list(SEVERITY_LEVELS)}")

    rows = []
    t_start = time.time()
    n_configs = (
        len(models) * len(categories) * len(corruption_types) * len(severity_levels)
    )
    config_idx = 0

    for model_name in models:
        if verbose:
            print(f"\n[loading {model_name}]")
        classifier = load_classifier(model_name, device=device)

        for category in categories:
            samples = collect_test_samples(category, data_root)
            prompts = prompts_all[category][strategy]

            for corruption_type in corruption_types:
                for severity in severity_levels:
                    config_idx += 1
                    np.random.seed(seed)  # for noise-based corruptions

                    try:
                        # Build corrupted sample list — dicts still, but with
                        # image_path replaced by an in-memory PIL image handle
                        # via a thin wrapper class isn't needed because
                        # zero_shot_classify_category opens paths. So we
                        # inline the corruption + inference here.
                        results = _classify_corrupted(
                            classifier, samples, prompts,
                            corruption_type, severity,
                            batch_size=batch_size,
                        )
                        metrics = compute_metrics(results)
                        row = {
                            "model": model_name,
                            "category": category,
                            "corruption": corruption_type,
                            "severity": severity,
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
                        }
                        rows.append(row)

                        # Per-run checkpoint (matches your Week 3 pattern)
                        pd.DataFrame(rows).to_csv(output_path, index=False)

                        if verbose:
                            elapsed = time.time() - t_start
                            print(
                                f"[{elapsed:6.1f}s] [{config_idx}/{n_configs}] "
                                f"{model_name:>6} | {category:<12} | "
                                f"{corruption_type:<18} | sev={severity} | "
                                f"bal_acc={metrics['balanced_accuracy']:.3f}  "
                                f"AUROC={metrics['auroc']:.3f}"
                            )
                    except Exception as e:
                        print(
                            f"[ERROR] {model_name} | {category} | "
                            f"{corruption_type} | sev={severity}: {e}"
                        )

        del classifier

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nDone. {len(df)} rows saved to {output_path}")
        print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")

    return df


def _classify_corrupted(
    classifier,
    samples: list[dict[str, str]],
    prompts: list[str],
    corruption_type: str,
    severity: int,
    batch_size: int = 16,
) -> list[dict]:
    """Run inference on corrupted images. Internal helper for corruption sweep."""
    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        images = []
        for s in batch:
            img = Image.open(s["image_path"]).convert("RGB")
            img = corrupt_image(img, corruption_type, severity)
            images.append(img)

        probs = classifier.classify_images(images, prompts)

        for sample, prob in zip(batch, probs):
            prob_good, prob_defective = float(prob[0]), float(prob[1])
            predicted = "good" if prob_good > prob_defective else "defective"
            results.append({
                "image_path": sample["image_path"],
                "true_label": sample["true_label"],
                "subtype": sample["subtype"],
                "predicted_label": predicted,
                "prob_good": prob_good,
                "prob_defective": prob_defective,
            })
    return results

def run_threshold_calibration_sweep(
    prompts_config: str | Path,
    data_root: str | Path,
    models: list[str],
    categories: list[str] | str,
    strategy: str,
    k_calib: int,
    seeds: list[int],
    output_path: str | Path,
    batch_size: int = 16,
    device: str = "cuda",
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate threshold-calibrated zero-shot classification.

    For each (model, category, seed), samples k_calib good + k_calib defective
    for a calibration set, finds the optimal decision threshold, then
    evaluates on the disjoint remaining test images with that threshold.
    This is a training-free baseline that isolates the effect of decision
    threshold choice from any representation-learning effect.

    Returns per-run metrics: n_calib, chosen_threshold, calib_balacc, and
    all standard eval metrics computed at that threshold.
    """
    prompts_all = load_prompts(prompts_config)

    if categories == "all":
        categories = sorted(prompts_all.keys())
    elif isinstance(categories, str):
        raise ValueError(f"categories must be 'all' or a list, got string {categories!r}")

    rows = []
    t_start = time.time()

    for model_name in models:
        if verbose:
            print(f"\n[loading {model_name}]")
        classifier = load_classifier(model_name, device=device)

        for category in categories:
            samples = collect_test_samples(category, data_root)
            prompts = prompts_all[category][strategy]

            for seed in seeds:
                try:
                    # Three-way split with k_train=0 (we don't train here)
                    _, calib_samples, eval_samples = sample_three_way_split(
                        samples, k_train=0, k_calib=k_calib, seed=seed,
                    )

                    # Zero-shot inference on calibration set
                    calib_results = zero_shot_classify_category(
                        classifier, calib_samples, prompts, batch_size=batch_size,
                    )
                    calib_scores = [r["prob_defective"] for r in calib_results]
                    calib_labels = [1 if r["true_label"] == "defective" else 0
                                    for r in calib_results]

                    # Find optimal threshold
                    threshold, calib_balacc = find_optimal_threshold(
                        calib_scores, calib_labels,
                    )

                    # Zero-shot inference on eval set, then recompute predictions
                    eval_results_raw = zero_shot_classify_category(
                        classifier, eval_samples, prompts, batch_size=batch_size,
                    )
                    eval_results = apply_threshold_to_results(
                        eval_results_raw, threshold=threshold,
                    )
                    metrics = compute_metrics(eval_results)

                    row = {
                        "model": model_name,
                        "category": category,
                        "seed": seed,
                        "strategy": strategy,
                        "k_calib": k_calib,
                        "chosen_threshold": threshold,
                        "calib_balacc": calib_balacc,
                        "n_calib": len(calib_samples),
                        "n_eval": metrics["n_total"],
                        "n_eval_good": metrics["n_good"],
                        "n_eval_defective": metrics["n_defective"],
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "auroc": metrics["auroc"],
                        "confusion_matrix": str(metrics["confusion_matrix"]),
                    }
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output_path, index=False)

                    if verbose:
                        elapsed = time.time() - t_start
                        print(
                            f"[{elapsed:6.1f}s] {model_name:>6} | {category:<12} | "
                            f"seed={seed} | thresh={threshold:.3f} | "
                            f"bal_acc={metrics['balanced_accuracy']:.3f} | "
                            f"AUROC={metrics['auroc']:.3f}"
                        )
                except Exception as e:
                    print(f"[ERROR] {model_name} | {category} | seed={seed}: {e}")

        del classifier

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nDone. {len(df)} rows saved to {output_path}")
        print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")

    return df


def run_lora_corruption_sweep(
    prompts_config: str | Path,
    data_root: str | Path,
    adapter_dir: str | Path,
    models: list[str],
    categories: list[str] | str,
    strategy: str,
    k: int,
    seed: int,
    corruption_types: list[str],
    severity_levels: list[int],
    output_path: str | Path,
    batch_size: int = 16,
    device: str = "cuda",
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate LoRA-adapted models under image corruption.

    For each (model, category, corruption, severity), loads the pre-trained
    LoRA adapter for that (model, category, k, seed) combination and
    evaluates on corrupted images. Adapters must exist at
    {adapter_dir}/{model}_{category}_k{k}_seed{seed}/.

    This tests whether few-shot adaptation improvements survive under
    realistic image degradations, or whether adaptation trades robustness
    for clean accuracy.
    """
    from src.corruptions import corrupt_image
    from PIL import Image

    prompts_all = load_prompts(prompts_config)
    if categories == "all":
        categories = sorted(prompts_all.keys())
    elif isinstance(categories, str):
        raise ValueError(f"categories must be 'all' or a list, got {categories!r}")

    adapter_dir = Path(adapter_dir)

    rows = []
    t_start = time.time()
    n_configs = (
        len(models) * len(categories) * len(corruption_types) * len(severity_levels)
    )
    config_idx = 0

    for model_name in models:
        for category in categories:
            samples = collect_test_samples(category, data_root)

            # We need the SAME train/eval split used during training,
            # so LoRA is evaluated on unseen images. Recreate deterministically.
            train_samples, eval_samples = sample_few_shot_split(
                samples, k=k, seed=seed,
            )

            prompts = prompts_all[category][strategy]
            adapter_path = adapter_dir / f"{model_name}_{category}_k{k}_seed{seed}"

            if not adapter_path.is_dir():
                print(f"[SKIP] Adapter not found: {adapter_path}")
                continue

            # Load fresh classifier + trained adapter
            classifier = load_classifier(model_name, device=device)
            load_adapter(classifier.model, adapter_path)
            classifier.model.eval()

            for corruption_type in corruption_types:
                for severity in severity_levels:
                    config_idx += 1
                    np.random.seed(seed)

                    try:
                        results = _classify_corrupted(
                            classifier, eval_samples, prompts,
                            corruption_type, severity, batch_size=batch_size,
                        )
                        metrics = compute_metrics(results)

                        row = {
                            "model": model_name,
                            "category": category,
                            "k": k,
                            "seed": seed,
                            "corruption": corruption_type,
                            "severity": severity,
                            "strategy": strategy,
                            "n_eval": metrics["n_total"],
                            "n_eval_good": metrics["n_good"],
                            "n_eval_defective": metrics["n_defective"],
                            "accuracy": metrics["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "f1": metrics["f1"],
                            "auroc": metrics["auroc"],
                            "confusion_matrix": str(metrics["confusion_matrix"]),
                        }
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(output_path, index=False)

                        if verbose:
                            elapsed = time.time() - t_start
                            print(
                                f"[{elapsed:6.1f}s] [{config_idx}/{n_configs}] "
                                f"{model_name:>6} | {category:<12} | "
                                f"{corruption_type:<18} | sev={severity} | "
                                f"bal_acc={metrics['balanced_accuracy']:.3f}"
                            )
                    except Exception as e:
                        print(
                            f"[ERROR] {model_name} | {category} | "
                            f"{corruption_type} | sev={severity}: {e}"
                        )

            del classifier
            import torch
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\nDone. {len(df)} rows saved to {output_path}")
        print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")

    return df