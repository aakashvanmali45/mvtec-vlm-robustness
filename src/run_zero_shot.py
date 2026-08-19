"""CLI entry point for the zero-shot classification sweep.

Usage:
    python scripts/run_zero_shot.py --config configs/zero_shot.yaml
    python scripts/run_zero_shot.py --config configs/zero_shot.yaml --data-root /path/to/mvtec-ad
    python scripts/run_zero_shot.py --config configs/zero_shot.yaml --categories bottle,cable
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Ensure repo root is importable when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import run_zero_shot_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot VLM classification sweep on MVTec-AD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override data_root from the config (useful for Kaggle vs local paths).",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated categories to run (overrides config). "
             "Useful for smoke tests, e.g. 'bottle,cable'.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Override output_path from the config.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device ('cuda' or 'cpu'). Auto-detected if not given.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-configuration progress prints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.is_file():
        sys.exit(f"Config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides on top of config.
    data_root = args.data_root or config["data_root"]
    output_path = args.output or config["output_path"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        categories = config["categories"]

    print(f"Experiment:   {config.get('experiment_name', '(unnamed)')}")
    print(f"Config:       {args.config}")
    print(f"Data root:    {data_root}")
    print(f"Device:       {device}")
    print(f"Models:       {config['models']}")
    print(f"Strategies:   {config['strategies']}")
    print(f"Categories:   {categories if categories != 'all' else 'all (from prompts config)'}")
    print(f"Output:       {output_path}")

    run_zero_shot_sweep(
        prompts_config=config["prompts_config"],
        data_root=data_root,
        models=config["models"],
        strategies=config["strategies"],
        categories=categories,
        output_path=output_path,
        batch_size=config.get("batch_size", 16),
        device=device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()