"""LoRA fine-tuning for CLIP on MVTec-AD binary classification.

Trains a per-category LoRA adapter using a small labeled set (k good + k defective
images). The base CLIP weights stay frozen; only LoRA adapters on the vision
encoder's attention layers are trained.

Training objective: contrastive image-text matching. For each training image,
maximize similarity to its true prompt (good or defective) and minimize
similarity to the other prompt. This mirrors CLIP's original pretraining
objective on a two-class problem.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from src.data import collect_test_samples
from src.models import CLIPClassifier, load_classifier


# ------------------- Configuration -------------------

@dataclass
class LoRATrainingConfig:
    """Hyperparameters for LoRA fine-tuning. Kept as a dataclass for clarity
    and so it can be easily instantiated from a YAML config."""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 20
    batch_size: int = 8


# ------------------- Data sampling -------------------

def sample_few_shot_split(
    samples: list[dict[str, str]],
    k: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split test samples into a few-shot training set and an evaluation set.

    Selects k 'good' and k 'defective' images uniformly at random for training;
    everything else becomes the evaluation set.

    Args:
        samples: Output of src.data.collect_test_samples.
        k: Number of examples per class to select for training.
        seed: RNG seed for reproducibility.

    Returns:
        (train_samples, eval_samples). train_samples has exactly 2k entries.

    Raises:
        ValueError: If either class has fewer than k available samples.
    """
    rng = random.Random(seed)

    good = [s for s in samples if s["true_label"] == "good"]
    defective = [s for s in samples if s["true_label"] == "defective"]

    if len(good) < k:
        raise ValueError(f"Only {len(good)} good samples available, need {k}")
    if len(defective) < k:
        raise ValueError(f"Only {len(defective)} defective samples available, need {k}")

    good_train_idx = set(rng.sample(range(len(good)), k))
    def_train_idx = set(rng.sample(range(len(defective)), k))

    train = (
        [good[i] for i in good_train_idx] +
        [defective[i] for i in def_train_idx]
    )
    evaluation = (
        [g for i, g in enumerate(good) if i not in good_train_idx] +
        [d for i, d in enumerate(defective) if i not in def_train_idx]
    )

    return train, evaluation


# ------------------- Dataset -------------------

class FewShotDataset(Dataset):
    """PyTorch Dataset yielding (image, label_index) pairs.

    label_index is 0 for good, 1 for defective — matches the prompt convention.
    """

    def __init__(self, samples: list[dict[str, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        label = 0 if s["true_label"] == "good" else 1
        return img, label


def collate_fn(batch: list[tuple[Image.Image, int]]) -> tuple[list[Image.Image], torch.Tensor]:
    """Custom collate: PIL images can't be default-collated, so we keep them as a list.
    Labels are stacked into a tensor."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


# ------------------- LoRA attach -------------------

def attach_lora(clip_model, config: LoRATrainingConfig):
    """Freeze the entire CLIP model, then attach LoRA adapters to the vision
    encoder's attention layers. Only the adapter weights become trainable.

    Modifies clip_model in place and also returns it for convenience.
    """
    # Freeze everything first — explicit, not trusting library defaults.
    for param in clip_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.dropout,
        bias="none",
    )
    clip_model.vision_model = get_peft_model(clip_model.vision_model, lora_config)

    return clip_model


# ------------------- Training loop -------------------

def train_lora_adapter(
    classifier: CLIPClassifier,
    train_samples: list[dict[str, str]],
    prompts: Sequence[str],
    config: LoRATrainingConfig,
    device: str = "cuda",
    verbose: bool = True,
) -> list[float]:
    """Train a LoRA adapter on the CLIP vision encoder for one category.

    Uses contrastive image-text matching: for each image, compute similarity to
    both prompts, and cross-entropy against the true label index.

    Args:
        classifier: A CLIPClassifier (must already have LoRA attached via attach_lora).
        train_samples: The k+k few-shot training set.
        prompts: [good_prompt, defective_prompt].
        config: Training hyperparameters.
        device: Torch device.
        verbose: If True, print per-epoch loss.

    Returns:
        List of average epoch losses, one per epoch.
    """
    if len(prompts) != 2:
        raise ValueError(f"Expected 2 prompts, got {len(prompts)}")

    model = classifier.model
    processor = classifier.processor

    # Precompute text inputs — prompts don't change across the training loop.
    text_inputs = processor(text=list(prompts), return_tensors="pt", padding=True).to(device)

    dataset = FewShotDataset(train_samples)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Only pass trainable (LoRA) params to the optimizer. Frozen ones don't need it.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.train()
    epoch_losses = []

    for epoch in range(config.epochs):
        batch_losses = []
        for images, labels in loader:
            labels = labels.to(device)
            image_inputs = processor(images=images, return_tensors="pt").to(device)

            # Forward: get logits_per_image, which is (batch, 2) similarity scores.
            outputs = model(
                pixel_values=image_inputs["pixel_values"],
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            logits = outputs.logits_per_image  # (batch, 2)

            # Cross-entropy: labels are 0 (good) or 1 (defective), matching prompt indices.
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = float(np.mean(batch_losses))
        epoch_losses.append(avg_loss)

        if verbose:
            print(f"  epoch {epoch+1:3d}/{config.epochs} — loss: {avg_loss:.4f}")

    model.eval()
    return epoch_losses


# ------------------- Reproducibility -------------------

def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs. Should be called at the start of
    every training run for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)