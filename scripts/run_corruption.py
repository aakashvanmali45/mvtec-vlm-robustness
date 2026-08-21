"""CLI entry point for the corruption robustness evaluation.

Usage:
    python scripts/run_corruption.py --config configs/corruption.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import run_corruption_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot corruption robustness evaluation on MVTec-AD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories (overrides config).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.is_file():
        sys.exit(f"Config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = args.data_root or config["data_root"]
    output_path = args.output or config["output_path"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    else:
        categories = config["categories"]

    print(f"Experiment:        {config.get('experiment_name', '(unnamed)')}")
    print(f"Config:            {args.config}")
    print(f"Data root:         {data_root}")
    print(f"Device:            {device}")
    print(f"Models:            {config['models']}")
    print(f"Categories:        {categories if categories != 'all' else 'all (from prompts config)'}")
    print(f"Strategy:          {config['strategy']}")
    print(f"Corruption types:  {config['corruption_types']}")
    print(f"Severity levels:   {config['severity_levels']}")
    print(f"Output:            {output_path}")

    run_corruption_evaluation(
        prompts_config=config["prompts_config"],
        data_root=data_root,
        models=config["models"],
        categories=categories,
        strategy=config["strategy"],
        fewshot_adapter_dir=None,  # zero-shot for now
        output_path=output_path,
        corruption_types=config["corruption_types"],
        severity_levels=config["severity_levels"],
        batch_size=config.get("batch_size", 16),
        device=device,
        seed=config.get("seed", 42),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()