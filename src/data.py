"""Data loading utilities for MVTec-AD experiments.

Two responsibilities:
1. Load prompt configurations from YAML.
2. Enumerate test images for a given MVTec-AD category with their labels.
"""

from pathlib import Path
from typing import Any

import yaml


# Recognized image file extensions in MVTec-AD.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_prompts(config_path: str | Path) -> dict[str, dict[str, list[str]]]:
    """Load prompt configurations from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Nested dict: {category: {strategy: [good_prompt, defective_prompt]}}.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config structure is malformed.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    if not isinstance(prompts, dict):
        raise ValueError(f"Expected top-level dict in {config_path}, got {type(prompts)}")

    # Validate structure: each category has strategies, each strategy is a list of exactly 2 strings.
    for category, strategies in prompts.items():
        if not isinstance(strategies, dict):
            raise ValueError(f"Category '{category}' must map to a dict of strategies")
        for strategy, prompt_list in strategies.items():
            if not isinstance(prompt_list, list) or len(prompt_list) != 2:
                raise ValueError(
                    f"{category}.{strategy} must be a list of exactly 2 prompts, "
                    f"got {prompt_list!r}"
                )
            if not all(isinstance(p, str) for p in prompt_list):
                raise ValueError(f"{category}.{strategy} must contain strings only")

    return prompts


def collect_test_samples(
    category: str,
    data_root: str | Path,
) -> list[dict[str, str]]:
    """Collect all test images for one MVTec-AD category with binary labels.

    Iterates test/*/ subdirectories. Images under test/good/ are labeled 'good';
    everything else is labeled 'defective'.

    Args:
        category: MVTec-AD category name (e.g. 'bottle').
        data_root: Path to the MVTec-AD dataset root (containing 15 category folders).

    Returns:
        List of dicts with keys 'image_path', 'true_label', 'subtype'.
        'subtype' is the immediate subdirectory name (e.g. 'good', 'broken_large').

    Raises:
        FileNotFoundError: If the category's test directory does not exist.
        RuntimeError: If no images are found.
    """
    test_root = Path(data_root) / category / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    samples = []
    for subdir in sorted(test_root.iterdir()):
        if not subdir.is_dir():
            continue
        true_label = "good" if subdir.name == "good" else "defective"
        for img_file in sorted(subdir.iterdir()):
            if img_file.suffix.lower() in _IMAGE_EXTENSIONS:
                samples.append({
                    "image_path": str(img_file),
                    "true_label": true_label,
                    "subtype": subdir.name,
                })

    if not samples:
        raise RuntimeError(f"No test images found under {test_root}")

    return samples