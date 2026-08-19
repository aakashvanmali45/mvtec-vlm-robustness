"""Tests for src/models.py."""

import numpy as np
import pytest
from PIL import Image

from src.models import (
    MODEL_REGISTRY,
    ZeroShotClassifier,
    load_classifier,
    zero_shot_classify_category,
)


# ------------------- Tests for load_classifier -------------------

def test_load_classifier_unknown_name_raises():
    """Unknown model names should raise ValueError with a helpful message."""
    with pytest.raises(ValueError, match="Unknown model"):
        load_classifier("does-not-exist", device="cpu")


def test_model_registry_populated():
    """MODEL_REGISTRY should contain at least clip and siglip."""
    assert "clip" in MODEL_REGISTRY
    assert "siglip" in MODEL_REGISTRY


# ------------------- Tests for zero_shot_classify_category -------------------

class DummyClassifier(ZeroShotClassifier):
    """Test double that returns deterministic probs without needing a real model.

    Predicts high probability for whichever prompt index equals the true label:
    - if image's 'true_label_hint' attribute is 'good', returns [0.9, 0.1]
    - otherwise returns [0.1, 0.9]
    """

    def __init__(self):
        # Skip parent __init__ to avoid needing a real model/processor.
        self.device = "cpu"

    def classify_images(self, images, prompts):
        # Read hint from the first pixel of each fake image (see fixture below).
        n = len(images)
        probs = np.zeros((n, 2), dtype=np.float32)
        for i, img in enumerate(images):
            # Hint encoded in the top-left pixel: (0,0,0) = good, (255,0,0) = defective
            r = img.getpixel((0, 0))[0]
            if r == 0:
                probs[i] = [0.9, 0.1]
            else:
                probs[i] = [0.1, 0.9]
        return probs


def _make_hint_image(is_good: bool, tmp_path, filename: str) -> str:
    """Create a tiny image whose top-left pixel encodes 'good' or 'defective'."""
    color = (0, 0, 0) if is_good else (255, 0, 0)
    img = Image.new("RGB", (10, 10), color)
    path = tmp_path / filename
    img.save(path)
    return str(path)


def test_zero_shot_classify_category_end_to_end(tmp_path):
    """DummyClassifier + fake samples should produce correct predictions."""
    samples = [
        {"image_path": _make_hint_image(True,  tmp_path, "good1.png"),
         "true_label": "good", "subtype": "good"},
        {"image_path": _make_hint_image(True,  tmp_path, "good2.png"),
         "true_label": "good", "subtype": "good"},
        {"image_path": _make_hint_image(False, tmp_path, "def1.png"),
         "true_label": "defective", "subtype": "broken"},
    ]

    classifier = DummyClassifier()
    prompts = ["a photo of a good X", "a photo of a defective X"]
    results = zero_shot_classify_category(classifier, samples, prompts, batch_size=2)

    assert len(results) == 3
    for r in results:
        assert r["predicted_label"] == r["true_label"]
        assert r["prob_good"] + r["prob_defective"] == pytest.approx(1.0, abs=1e-5)
    # subtype is preserved from samples
    subtypes = [r["subtype"] for r in results]
    assert subtypes == ["good", "good", "broken"]


def test_zero_shot_classify_category_wrong_prompt_count_raises(tmp_path):
    """Passing !=2 prompts should raise ValueError."""
    classifier = DummyClassifier()
    samples = [{
        "image_path": _make_hint_image(True, tmp_path, "x.png"),
        "true_label": "good", "subtype": "good"
    }]

    with pytest.raises(ValueError, match="Expected 2 prompts"):
        zero_shot_classify_category(classifier, samples, ["only one"], batch_size=1)

    with pytest.raises(ValueError, match="Expected 2 prompts"):
        zero_shot_classify_category(classifier, samples, ["a", "b", "c"], batch_size=1)


# ------------------- Optional integration test (requires model download) -------------------

@pytest.mark.slow
def test_clip_classifier_runs_on_real_model(tmp_path):
    """Integration test: verify CLIP loads and classifies without error.

    Skipped by default because it downloads ~600MB. Run with:
        pytest -m slow
    """
    classifier = load_classifier("clip", device="cpu")
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    probs = classifier.classify_images([img], ["a photo of a cat", "a photo of a dog"])
    assert probs.shape == (1, 2)
    assert probs[0].sum() == pytest.approx(1.0, abs=1e-4)