"""Tests for src/data.py."""

from pathlib import Path

import pytest

from src.data import collect_test_samples, load_prompts


# ------------------- Tests for load_prompts -------------------

def test_load_prompts_returns_all_15_categories(tmp_path):
    """A well-formed YAML with 15 categories should load without error."""
    # Write a minimal well-formed prompts config with 15 categories.
    categories = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
    ]
    yaml_content = ""
    for cat in categories:
        yaml_content += f"""
{cat}:
  naive:
    - "good {cat}"
    - "bad {cat}"
  visual_primitive:
    - "good visual {cat}"
    - "bad visual {cat}"
  category_specific:
    - "good specific {cat}"
    - "bad specific {cat}"
"""
    config_file = tmp_path / "prompts.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    prompts = load_prompts(config_file)

    assert len(prompts) == 15
    for cat in categories:
        assert cat in prompts
        assert set(prompts[cat].keys()) == {"naive", "visual_primitive", "category_specific"}
        for strategy in prompts[cat]:
            assert len(prompts[cat][strategy]) == 2


def test_load_prompts_missing_file_raises():
    """Non-existent config path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_prompts("nonexistent_file.yaml")


def test_load_prompts_malformed_raises(tmp_path):
    """A strategy with only one prompt should raise ValueError."""
    bad_yaml = """
bottle:
  naive:
    - "only one prompt"
"""
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(bad_yaml, encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 2 prompts"):
        load_prompts(config_file)


def test_load_actual_project_prompts():
    """The real configs/prompts.yaml should load and contain all 15 categories x 3 strategies."""
    config_path = Path(__file__).parent.parent / "configs" / "prompts.yaml"
    if not config_path.is_file():
        pytest.skip("configs/prompts.yaml not found (skipping integration test)")

    prompts = load_prompts(config_path)
    assert len(prompts) == 15
    for cat, strategies in prompts.items():
        assert set(strategies.keys()) == {"naive", "visual_primitive", "category_specific"}


# ------------------- Tests for collect_test_samples -------------------

def test_collect_samples_missing_category_raises(tmp_path):
    """A non-existent category directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        collect_test_samples("nonexistent_category", tmp_path)


def test_collect_samples_correct_labels_and_subtypes(tmp_path):
    """Fake MVTec-AD structure: verify labels and subtypes are correctly extracted."""
    # Build a minimal fake structure: bottle/test/good/*.png and bottle/test/broken/*.png
    bottle_test = tmp_path / "bottle" / "test"
    (bottle_test / "good").mkdir(parents=True)
    (bottle_test / "broken").mkdir(parents=True)

    # Create 3 good images and 2 broken images (as empty files)
    for i in range(3):
        (bottle_test / "good" / f"{i:03d}.png").touch()
    for i in range(2):
        (bottle_test / "broken" / f"{i:03d}.png").touch()

    samples = collect_test_samples("bottle", tmp_path)

    assert len(samples) == 5
    assert sum(1 for s in samples if s["true_label"] == "good") == 3
    assert sum(1 for s in samples if s["true_label"] == "defective") == 2
    assert sum(1 for s in samples if s["subtype"] == "good") == 3
    assert sum(1 for s in samples if s["subtype"] == "broken") == 2


def test_collect_samples_ignores_non_image_files(tmp_path):
    """Non-image files (e.g. license.txt, ground_truth masks not in test/) are ignored."""
    bottle_test = tmp_path / "bottle" / "test" / "good"
    bottle_test.mkdir(parents=True)

    (bottle_test / "001.png").touch()
    (bottle_test / "002.jpg").touch()
    (bottle_test / "readme.txt").touch()  # should be ignored
    (bottle_test / "notes.md").touch()    # should be ignored

    samples = collect_test_samples("bottle", tmp_path.parent.parent if False else tmp_path)
    # (data_root = tmp_path, so it looks for tmp_path/bottle/test/*)

    assert len(samples) == 2
    for s in samples:
        assert s["image_path"].endswith((".png", ".jpg"))