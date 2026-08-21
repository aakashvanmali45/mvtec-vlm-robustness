"""CLI entry point for threshold calibration sweep."""

import argparse
import sys
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiment import run_threshold_calibration_sweep


def main():
    parser = argparse.ArgumentParser(description="Threshold calibration sweep.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = args.data_root or config["data_root"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_threshold_calibration_sweep(
        prompts_config=config["prompts_config"],
        data_root=data_root,
        models=config["models"],
        categories=config["categories"],
        strategy=config["strategy"],
        k_calib=config["k_calib"],
        seeds=config["seeds"],
        output_path=config["output_path"],
        batch_size=config.get("batch_size", 16),
        device=device,
    )


if __name__ == "__main__":
    main()