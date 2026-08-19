"""Experiment runners: orchestrate model sweeps over categories and strategies."""

import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import collect_test_samples, load_prompts
from src.metrics import compute_metrics
from src.models import load_classifier, zero_shot_classify_category


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