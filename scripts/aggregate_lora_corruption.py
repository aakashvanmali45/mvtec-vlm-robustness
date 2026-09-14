"""Aggregate multi-seed LoRA-under-corruption results into paper-ready tables.

Reads the per-run CSV produced by scripts/run_lora_corruption.py (with a
multiseed config) and emits:

  * per-model, per-corruption mean and std across categories x seeds x severities
    (drop-in replacement for the paper's Table 3)
  * per-severity breakdown, for the appendix
  * per-category tables, for the supplementary material (Reviewer 2 Comment 5)

Usage:
    python scripts/aggregate_lora_corruption.py \\
        --input  results/lora_corruption_multiseed_results.csv \\
        --outdir results/tables/lora_corruption

If --clean-baselines is passed with a CSV of clean LoRA k=10 results (the
output of run_few_shots.py at k=10), the script also emits a comparison
table showing the drop from clean to worst-corruption per (model, seed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLS = ["balanced_accuracy", "auroc"]


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "model", "category", "seed", "corruption", "severity",
        "balanced_accuracy", "auroc",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    return df


def table_per_corruption(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of each metric across categories x seeds x severities,
    grouped by (model, corruption). This is the paper's Table 3 shape."""
    grouped = df.groupby(["model", "corruption"])[METRIC_COLS]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    table = pd.concat([mean, std], axis=1).reset_index()
    # Add a "Mean" row per model across all corruptions.
    mean_rows = []
    for model, sub in df.groupby("model"):
        row = {"model": model, "corruption": "MEAN"}
        for c in METRIC_COLS:
            row[f"{c}_mean"] = sub[c].mean()
            row[f"{c}_std"] = sub[c].std(ddof=1)
        mean_rows.append(row)
    table = pd.concat([table, pd.DataFrame(mean_rows)], ignore_index=True)
    return table.sort_values(["model", "corruption"]).reset_index(drop=True)


def table_per_corruption_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std across categories x seeds, grouped by (model, corruption, severity).
    For the appendix."""
    grouped = df.groupby(["model", "corruption", "severity"])[METRIC_COLS]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    table = pd.concat([mean, std], axis=1).reset_index()
    return table.sort_values(["model", "corruption", "severity"]).reset_index(drop=True)


def table_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category mean across seeds x severities, per (model, category, corruption).
    For supplementary material."""
    grouped = df.groupby(["model", "category", "corruption"])[METRIC_COLS]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    table = pd.concat([mean, std], axis=1).reset_index()
    return table.sort_values(["model", "category", "corruption"]).reset_index(drop=True)


def table_worst_corruption(df: pd.DataFrame,
                            clean_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """For each (model, seed), find the worst per-metric corruption
    (mean across categories x severities). If clean baselines are provided,
    also report the drop from clean."""
    per_seed = (
        df.groupby(["model", "seed", "corruption"])[METRIC_COLS]
          .mean()
          .reset_index()
    )
    rows = []
    for (model, seed), sub in per_seed.groupby(["model", "seed"]):
        row = {"model": model, "seed": seed}
        for c in METRIC_COLS:
            worst_idx = sub[c].idxmin()
            row[f"worst_{c}_corruption"] = sub.loc[worst_idx, "corruption"]
            row[f"worst_{c}_value"] = sub.loc[worst_idx, c]
        rows.append(row)
    out = pd.DataFrame(rows)

    if clean_df is not None:
        clean_per_seed = (
            clean_df[clean_df["k"] == clean_df["k"].max()]
                .groupby(["model", "seed"])[METRIC_COLS]
                .mean()
                .reset_index()
                .rename(columns={c: f"clean_{c}" for c in METRIC_COLS})
        )
        out = out.merge(clean_per_seed, on=["model", "seed"], how="left")
        for c in METRIC_COLS:
            out[f"drop_{c}"] = out[f"clean_{c}"] - out[f"worst_{c}_value"]

    return out.sort_values(["model", "seed"]).reset_index(drop=True)


def format_table_3(per_corruption: pd.DataFrame) -> pd.DataFrame:
    """Emit a wide-format LaTeX-friendly table matching Table 3 in the paper:
    rows = corruption, columns = (model, bal_acc mean+std | auroc mean+std)."""
    wide = per_corruption.pivot(index="corruption", columns="model")
    # Order corruptions the way the paper does; MEAN last.
    order = [
        "brightness", "contrast", "gaussian_blur",
        "gaussian_noise", "jpeg_compression", "MEAN",
    ]
    wide = wide.reindex([c for c in order if c in wide.index])
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True,
                        help="Per-run CSV from run_lora_corruption.py")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Directory to write aggregated tables into")
    parser.add_argument("--clean-baselines", type=Path, default=None,
                        help="Optional few-shot results CSV for clean baseline comparison")
    args = parser.parse_args()

    df = load_results(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Table 3 replacement.
    t_corruption = table_per_corruption(df)
    t_corruption.to_csv(args.outdir / "table_per_corruption.csv", index=False)

    wide = format_table_3(t_corruption)
    wide.to_csv(args.outdir / "table_3_wide.csv")

    # Appendix / supplement.
    table_per_corruption_severity(df).to_csv(
        args.outdir / "table_per_corruption_severity.csv", index=False
    )
    table_per_category(df).to_csv(
        args.outdir / "table_per_category.csv", index=False
    )

    # Worst-corruption summary, optionally with clean baseline drops.
    clean_df = None
    if args.clean_baselines is not None:
        clean_df = pd.read_csv(args.clean_baselines)
    table_worst_corruption(df, clean_df=clean_df).to_csv(
        args.outdir / "table_worst_corruption.csv", index=False
    )

    # Console summary — this is the number that goes in the paper prose.
    print("\n=== Per-model mean across ALL corruptions x severities x seeds x categories ===")
    for model, sub in df.groupby("model"):
        print(f"{model:>6}  bal_acc: {sub['balanced_accuracy'].mean():.3f} "
              f"± {sub['balanced_accuracy'].std(ddof=1):.3f}   "
              f"auroc: {sub['auroc'].mean():.3f} "
              f"± {sub['auroc'].std(ddof=1):.3f}   "
              f"(n = {len(sub)} runs across "
              f"{sub['seed'].nunique()} seeds x "
              f"{sub['category'].nunique()} categories x "
              f"{sub['corruption'].nunique()} corruptions x "
              f"{sub['severity'].nunique()} severities)")

    print(f"\nAll tables written to {args.outdir}")


if __name__ == "__main__":
    main()
