"""Model wrappers for zero-shot image-text classification.

Provides a unified interface across CLIP and SigLIP families, hiding
model-specific differences in scoring (softmax vs sigmoid), tokenizer padding,
and output type conventions.

Usage:
    classifier = load_classifier("clip", device="cuda")
    probs = classifier.classify_images(images, prompts)
    # probs[i, j] = probability that image i matches prompt j
"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor


# Registered model checkpoints. Add new ones here as the project grows.
MODEL_REGISTRY = {
    "clip":   "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-base-patch16-224",
}


class ZeroShotClassifier(ABC):
    """Abstract base for zero-shot image-text classifiers.

    Subclasses must implement classify_images. All subclasses share the same
    public interface so downstream code can swap models without changes.
    """

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    @abstractmethod
    def classify_images(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> np.ndarray:
        """Return a probability matrix of shape (n_images, n_prompts).

        Rows sum to 1. probs[i, j] is the probability that image i matches prompt j.
        """
        raise NotImplementedError


class CLIPClassifier(ZeroShotClassifier):
    """CLIP-family classifier using softmax across prompts.

    CLIP was trained with cross-prompt softmax loss, so probabilities are
    naturally comparable across prompts via softmax.
    """

    def classify_images(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> np.ndarray:
        inputs = self.processor(
            text=list(prompts),
            images=list(images),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # (n_images, n_prompts)
            probs = logits.softmax(dim=-1)

        return probs.cpu().numpy()


class SigLIPClassifier(ZeroShotClassifier):
    """SigLIP-family classifier using per-prompt sigmoid, then renormalized.

    SigLIP was trained with per-prompt sigmoid loss. Each prompt-image pair has
    an independent probability; softmax across prompts fabricates a comparison
    SigLIP was not trained for. We apply sigmoid per prompt, then renormalize
    so rows sum to 1 for downstream metric compatibility.
    """

    def classify_images(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> np.ndarray:
        inputs = self.processor(
            text=list(prompts),
            images=list(images),
            return_tensors="pt",
            padding="max_length",  # SigLIP requires fixed-length padding
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image  # (n_images, n_prompts)
            sig_probs = torch.sigmoid(logits)
            probs = sig_probs / sig_probs.sum(dim=-1, keepdim=True)

        return probs.cpu().numpy()


def load_classifier(model_name: str, device: str = "cuda") -> ZeroShotClassifier:
    """Factory: load and return a classifier for the given model name.

    Args:
        model_name: One of the keys in MODEL_REGISTRY (e.g. 'clip', 'siglip').
        device: Torch device string ('cuda' or 'cpu').

    Returns:
        A ready-to-use ZeroShotClassifier instance.

    Raises:
        ValueError: If model_name is not registered.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Registered: {sorted(MODEL_REGISTRY)}"
        )

    checkpoint = MODEL_REGISTRY[model_name]

    if model_name == "clip":
        model = CLIPModel.from_pretrained(checkpoint).to(device)
        processor = CLIPProcessor.from_pretrained(checkpoint)
        return CLIPClassifier(model, processor, device)

    if model_name == "siglip":
        model = AutoModel.from_pretrained(checkpoint).to(device)
        processor = AutoProcessor.from_pretrained(checkpoint)
        return SigLIPClassifier(model, processor, device)

    # Should be unreachable given the registry check above, but defensive.
    raise ValueError(f"No classifier implementation for '{model_name}'")

def zero_shot_classify_category(
    classifier: ZeroShotClassifier,
    samples: list[dict[str, str]],
    prompts: Sequence[str],
    batch_size: int = 16,
) -> list[dict]:
    """Run zero-shot classification on a list of samples.

    Args:
        classifier: A ZeroShotClassifier instance.
        samples: Output of src.data.collect_test_samples.
        prompts: List of exactly 2 prompts, [good_prompt, defective_prompt].
        batch_size: Images per forward pass.

    Returns:
        List of dicts with keys: image_path, true_label, subtype,
        predicted_label, prob_good, prob_defective.
    """
    if len(prompts) != 2:
        raise ValueError(f"Expected 2 prompts (good, defective), got {len(prompts)}")

    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        images = [Image.open(s["image_path"]).convert("RGB") for s in batch]
        probs = classifier.classify_images(images, prompts)  # (batch, 2)

        for sample, prob in zip(batch, probs):
            prob_good, prob_defective = float(prob[0]), float(prob[1])
            predicted = "good" if prob_good > prob_defective else "defective"
            results.append({
                "image_path": sample["image_path"],
                "true_label": sample["true_label"],
                "subtype": sample["subtype"],
                "predicted_label": predicted,
                "prob_good": prob_good,
                "prob_defective": prob_defective,
            })

    return results