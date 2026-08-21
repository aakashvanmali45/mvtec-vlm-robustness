"""CLI entry point for LoRA + corruption sweep."""

import argparse
import sys
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiment import run_lora_corruption_sweep


def main():
    parser = argparse.ArgumentParser(description="LoRA + corruption sweep.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = args.data_root or config["data_root"]
    adapter_dir = args.adapter_dir or config["adapter_dir"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_lora_corruption_sweep(
        prompts_config=config["prompts_config"],
        data_root=data_root,
        adapter_dir=adapter_dir,
        models=config["models"],
        categories=config["categories"],
        strategy=config["strategy"],
        k=config["k"],
        seed=config["seed"],
        corruption_types=config["corruption_types"],
        severity_levels=config["severity_levels"],
        output_path=config["output_path"],
        batch_size=config.get("batch_size", 16),
        device=device,
    )


if __name__ == "__main__":
    main()