"""CLI entry point for the few-shot LoRA fine-tuning sweep.

Usage:
    python scripts/run_few_shot.py --config configs/few_shot.yaml
    python scripts/run_few_shot.py --config configs/few_shot.yaml --data-root /path/to/mvtec-ad
    python scripts/run_few_shot.py --config configs/few_shot.yaml --categories bottle,cable
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import run_few_shot_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run few-shot LoRA fine-tuning sweep on MVTec-AD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the experiment YAML config.")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from the config.")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories (overrides config).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output_path from the config.")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device. Auto-detected if not given.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-configuration progress prints.")
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

    print(f"Experiment:   {config.get('experiment_name', '(unnamed)')}")
    print(f"Config:       {args.config}")
    print(f"Data root:    {data_root}")
    print(f"Device:       {device}")
    print(f"Models:       {config['models']}")
    print(f"Categories:   {categories if categories != 'all' else 'all (from prompts config)'}")
    print(f"k values:     {config['k_values']}")
    print(f"Seeds:        {config['seeds']}")
    print(f"Strategy:     {config['strategy']}")
    print(f"LoRA config:  {config['lora']}")
    print(f"Output:       {output_path}")

    run_few_shot_sweep(
        prompts_config=config["prompts_config"],
        data_root=data_root,
        models=config["models"],
        categories=categories,
        k_values=config["k_values"],
        seeds=config["seeds"],
        strategy=config["strategy"],
        output_path=output_path,
        lora_config=config["lora"],
        adapter_save_dir=config.get("adapter_save_dir"),
        device=device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()