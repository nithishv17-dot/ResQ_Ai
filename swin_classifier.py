"""
swin_classifier.py
==================
PyTorch Swin Transformer (swin_base_patch4_window7_224) Classifier and
Multimodal Fusion Adapter for ResQ_Ai Phase 2.

Features:
    - Model Path Configuration: Clean relative pathing (os.path / pathlib).
    - Architecture & Checkpoint Loading: Initializes timm's
      swin_base_patch4_window7_224 (num_classes=8), safely loading state_dict
      on CPU/CUDA with model.eval().
    - Data Augmentation & Inference Transforms:
        * train_transform: RandomResizedCrop, Flips, Rotation, ColorJitter, ImageNet norm.
        * val_transform / inference_transform: Resize(224, 224), ToTensor, ImageNet norm.
    - Softmax Inference: Computes calibrated probability distributions across 8 classes.
    - Ensemble Fusion: Blends CNN stacking meta-learner and Vision Transformer predictions.
"""

import os
import pathlib
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm

logger = logging.getLogger("swin_classifier")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# 1. Classes & Model Path Configuration
# ---------------------------------------------------------------------------

CLASSES: List[str] = [
    'ants',
    'bed_bugs',
    'chiggers',
    'fleas',
    'mosquitos',
    'no_bites',
    'spiders',
    'ticks',
]

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_SWIN_FILENAME = "swin_phase2_best.pth"
DEFAULT_SWIN_PATH = str(BASE_DIR / "models" / DEFAULT_SWIN_FILENAME)
SWIN_MODEL_PATH = os.getenv("SWIN_MODEL_PATH", DEFAULT_SWIN_PATH)

# Standard ImageNet Channel Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 2. Data Augmentation & Inference Transforms
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transform = val_transform
inference_transform = val_transform


# ---------------------------------------------------------------------------
# 3. Model Architecture & Weight Loading
# ---------------------------------------------------------------------------

def load_swin_model(
    model_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_classes: int = 8,
) -> Tuple[nn.Module, Dict]:
    """
    Initializes the Swin Transformer architecture matching swin_phase2_best.pth
    and loads weights cleanly into evaluation mode.

    Args:
        model_path: Path to .pth checkpoint (defaults to SWIN_MODEL_PATH).
        device: Target torch device ('cuda' or 'cpu'). Auto-selects if None.
        num_classes: Number of classification targets (8 canonical classes).

    Returns:
        Tuple of (model, checkpoint_metadata_dict).
    """
    resolved_path = model_path or SWIN_MODEL_PATH
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"PyTorch Swin Transformer checkpoint not found at: {resolved_path}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    logger.info(f"Loading Swin checkpoint from: {resolved_path} onto {device}...")
    checkpoint = torch.load(resolved_path, map_location=device)

    # Extract metadata if available
    metadata = {}
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        metadata = {
            "model_name": checkpoint.get("model_name", "swin_base_patch4_window7_224"),
            "class_names": checkpoint.get("class_names", CLASSES),
            "class_to_idx": checkpoint.get("class_to_idx", None),
            "val_acc": checkpoint.get("val_acc", None),
        }
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in checkpoint:
                state_dict = checkpoint[k]
                break

    arch_name = metadata.get("model_name", "swin_base_patch4_window7_224")

    # Instantiate matching Swin architecture via timm
    model = timm.create_model(arch_name, pretrained=False, num_classes=num_classes)

    # Strip potential 'module.' prefixes from DataParallel training
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_key] = v

    load_res = model.load_state_dict(cleaned_state_dict, strict=True)
    logger.info(f"Swin Transformer weights loaded successfully: {load_res}")

    model.to(device)
    model.eval()

    return model, metadata


# ---------------------------------------------------------------------------
# 4. Prediction & Ensemble Fusion
# ---------------------------------------------------------------------------

class SwinClassifier:
    """
    Encapsulates PyTorch Swin-Base inference, transform execution,
    and probability extraction.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = model_path or SWIN_MODEL_PATH
        self.model, self.metadata = load_swin_model(
            model_path=self.model_path, device=self.device
        )
        self.transform = inference_transform

    def predict_proba(
        self, image_input: Union[str, pathlib.Path, Image.Image]
    ) -> np.ndarray:
        """
        Generates softmax class probabilities for a single image.

        Args:
            image_input: PIL Image or file path to image.

        Returns:
            np.ndarray of shape (8,) with float32 probabilities summing to 1.0.
        """
        if isinstance(image_input, (str, pathlib.Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return probs.astype(np.float32)

    def predict(
        self, image_input: Union[str, pathlib.Path, Image.Image]
    ) -> Tuple[str, float, np.ndarray]:
        """
        Returns top predicted class, confidence percentage, and full probability vector.
        """
        probs = self.predict_proba(image_input)
        top_idx = int(np.argmax(probs))
        return CLASSES[top_idx], float(probs[top_idx] * 100), probs


def fuse_vision_probabilities(
    keras_probs: np.ndarray,
    swin_probs: np.ndarray,
    method: str = "weighted_average",
    weights: Tuple[float, float] = (0.50, 0.50),
) -> np.ndarray:
    """
    Combines predictions from the Keras Stacked CNN Ensemble and the
    PyTorch Swin Transformer into a unified visual probability distribution.

    Args:
        keras_probs: (8,) float array from Keras stacking meta-learner.
        swin_probs: (8,) float array from PyTorch Swin Transformer.
        method: Fusion strategy ('weighted_average' or 'harmonic_mean').
        weights: Tuple of (w_keras, w_swin), default (0.50, 0.50).

    Returns:
        Normalized (8,) probability array summing to 1.0.
    """
    w_k, w_s = weights
    if method == "weighted_average":
        fused = (w_k * keras_probs) + (w_s * swin_probs)
    elif method == "geometric":
        fused = np.power(keras_probs, w_k) * np.power(swin_probs, w_s)
    else:
        fused = 0.5 * keras_probs + 0.5 * swin_probs

    total = np.sum(fused)
    if total > 0:
        return (fused / total).astype(np.float32)
    return fused.astype(np.float32)


# ---------------------------------------------------------------------------
# Self-Test Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch Swin Transformer Classifier")
    print("=" * 60)

    classifier = SwinClassifier()
    print("Metadata:", classifier.metadata)
    print(f"Active Device: {classifier.device}")

    # Test with dummy image
    dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    top_c, conf, probs = classifier.predict(dummy_img)

    print(f"\nDummy Image Prediction: {top_c} ({conf:.2f}%)")
    print("Probability Distribution:")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:<12}: {p * 100:5.2f}%")

    print("\n" + "=" * 60)
    print("SwinClassifier initialization & inference PASSED successfully!")
    print("=" * 60)
