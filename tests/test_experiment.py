"""Tests for src/experiment.py."""

from pathlib import Path

import pytest

from src.experiment import run_zero_shot_sweep


def test_run_zero_shot_sweep_invalid_categories_string_raises(tmp_path):
    """Passing a non-'all' string for categories should raise ValueError."""
    # Create a minimal fake prompts config.
    prompts_yaml = tmp_path / "prompts.yaml"
    prompts_yaml.write_text(
        'bottle:\n  naive:\n    - "good bottle"\n    - "bad bottle"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be 'all' or a list"):
        run_zero_shot_sweep(
            prompts_config=prompts_yaml,
            data_root=tmp_path,
            models=["clip"],
            strategies=["naive"],
            categories="bottle",  # BUG: string instead of list or 'all'
            output_path=tmp_path / "out.csv",
            device="cpu",
        )


def test_run_zero_shot_sweep_missing_category_raises(tmp_path):
    """Requesting a category not in the prompts config should raise KeyError."""
    prompts_yaml = tmp_path / "prompts.yaml"
    prompts_yaml.write_text(
        'bottle:\n  naive:\n    - "good bottle"\n    - "bad bottle"\n',
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="cable"):
        run_zero_shot_sweep(
            prompts_config=prompts_yaml,
            data_root=tmp_path,
            models=["clip"],
            strategies=["naive"],
            categories=["cable"],  # not in prompts config
            output_path=tmp_path / "out.csv",
            device="cpu",
        )


def test_run_zero_shot_sweep_missing_strategy_raises(tmp_path):
    """Requesting a strategy that doesn't exist for a category should raise KeyError."""
    prompts_yaml = tmp_path / "prompts.yaml"
    prompts_yaml.write_text(
        'bottle:\n  naive:\n    - "good bottle"\n    - "bad bottle"\n',
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="visual_primitive"):
        run_zero_shot_sweep(
            prompts_config=prompts_yaml,
            data_root=tmp_path,
            models=["clip"],
            strategies=["visual_primitive"],  # not defined for bottle
            categories=["bottle"],
            output_path=tmp_path / "out.csv",
            device="cpu",
        )