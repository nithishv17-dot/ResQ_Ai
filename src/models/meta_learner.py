"""
src/models/meta_learner.py
==========================
Tri-Branch Stacking Meta-Learner & Multimodal Vision Fusion for ResQ_Ai Phase 2.

Integrates three distinct diagnostic vision paradigms:
1. Deep CNN Ensembles:
   `models/bug_bite_8class_stacked_model.keras` (Keras Tier-2 Stacked CNN)
2. Vision Transformers:
   `models/swin_phase2_best.pth` (PyTorch Swin-Base, 81.52% val acc)
3. Handcrafted Visual Descriptors:
   `models/descriptor_classifier.joblib` (Calibrated PCA-LogReg on color, texture, SIFT, morphology)

Provides:
- `fuse_tri_branch_probabilities`: Calibrated probability fusion.
- `TriBranchMetaLearner`: High-level inference manager combining all 3 models.
- `train_stacking_meta_learner`: Out-of-fold cross-validation meta-learner training.
"""

import os
import sys
import pathlib
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from PIL import Image
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from src.features.image_descriptors import BugBiteDescriptorExtractor
from src.models.descriptor_model import DescriptorClassifier
from swin_classifier import SwinClassifier, CLASSES, DEFAULT_SWIN_PATH

logger = logging.getLogger("TriBranchMetaLearner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_KERAS_PATH = str(BASE_DIR / "models" / "bug_bite_8class_stacked_model.keras")
DEFAULT_DESCRIPTOR_PATH = str(BASE_DIR / "models" / "descriptor_classifier.joblib")
DEFAULT_META_LEARNER_PATH = str(BASE_DIR / "models" / "tri_branch_meta_learner.joblib")

# Default calibration weights for weighted average fusion
# PyTorch Swin-B: 45% (highest single-model val acc 81.52%)
# Keras Stacked CNN: 40% (diverse multi-scale CNN representations)
# Handcrafted Descriptors: 15% (explainable biological/erythema priors)
DEFAULT_BRANCH_WEIGHTS = (0.45, 0.40, 0.15)

CHANNEL_MEANS = np.array([170.88408, 134.76701, 118.84133], dtype=np.float32)


def fuse_tri_branch_probabilities(
    swin_probs: np.ndarray,
    keras_probs: np.ndarray,
    desc_probs: np.ndarray,
    method: str = "weighted_average",
    weights: Tuple[float, float, float] = DEFAULT_BRANCH_WEIGHTS,
    meta_learner: Optional[Any] = None,
) -> np.ndarray:
    """
    Fuses 8-class probability vectors from the three vision branches.

    Args:
        swin_probs: (8,) PyTorch Swin-B probability distribution.
        keras_probs: (8,) Keras Stacked CNN probability distribution.
        desc_probs: (8,) Handcrafted Descriptor Classifier probability distribution.
        method: 'weighted_average' or 'stacking'.
        weights: (w_swin, w_keras, w_desc) tuple summing to 1.0.
        meta_learner: Optional fitted meta-classifier for stacking.

    Returns:
        np.ndarray: Fused probability distribution of shape (8,) summing to 1.0.
    """
    swin_p = np.asarray(swin_probs, dtype=np.float32)
    keras_p = np.asarray(keras_probs, dtype=np.float32)
    desc_p = np.asarray(desc_probs, dtype=np.float32)

    if method == "stacking" and meta_learner is not None:
        stacked_features = np.concatenate([swin_p, keras_p, desc_p]).reshape(1, -1)
        fused = meta_learner.predict_proba(stacked_features)[0].astype(np.float32)
    else:
        # Default: Normalized weighted average
        w_swin, w_keras, w_desc = weights
        total_w = w_swin + w_keras + w_desc
        w_swin /= total_w
        w_keras /= total_w
        w_desc /= total_w

        fused = (w_swin * swin_p) + (w_keras * keras_p) + (w_desc * desc_p)

    # Renormalize to ensure strict simplex constraint
    fused_sum = np.sum(fused)
    if fused_sum > 0:
        fused /= fused_sum
    else:
        fused = np.ones(len(CLASSES), dtype=np.float32) / len(CLASSES)

    return fused.astype(np.float32)


class TriBranchMetaLearner:
    """
    Inference orchestrator that loads all three diagnostic vision branches
    and executes tri-branch inference with visual XAI biomarkers extraction.
    """

    def __init__(
        self,
        swin_path: Optional[str] = None,
        keras_path: Optional[str] = None,
        descriptor_path: Optional[str] = None,
        meta_learner_path: Optional[str] = None,
        classes: List[str] = CLASSES,
    ):
        self.swin_path = swin_path or os.getenv("SWIN_MODEL_PATH", DEFAULT_SWIN_PATH)
        self.keras_path = keras_path or os.getenv("MODEL_PATH", DEFAULT_KERAS_PATH)
        self.descriptor_path = descriptor_path or os.getenv("DESCRIPTOR_MODEL_PATH", DEFAULT_DESCRIPTOR_PATH)
        self.meta_learner_path = meta_learner_path or os.getenv("META_LEARNER_PATH", DEFAULT_META_LEARNER_PATH)
        self.classes = list(classes)

        # 1. Swin Transformer
        self.swin_classifier = SwinClassifier(model_path=self.swin_path)

        # 2. Handcrafted Descriptor Classifier
        self.descriptor_classifier = DescriptorClassifier(
            model_path=self.descriptor_path,
            classes=self.classes,
        )

        # 3. Keras Model (lazy loaded or loaded on init)
        self.keras_model = None
        self._load_keras_model()

        # 4. Meta-Learner (optional stacking)
        self.meta_learner = None
        if os.path.exists(self.meta_learner_path):
            try:
                self.meta_learner = joblib.load(self.meta_learner_path)
                logger.info(f"Loaded Stacking Meta-Learner from {self.meta_learner_path}")
            except Exception as e:
                logger.warning(f"Could not load meta-learner from {self.meta_learner_path}: {e}")

    def _load_keras_model(self):
        """Loads the Keras Tier-2 Stacked CNN model."""
        if os.path.exists(self.keras_path):
            try:
                import tensorflow as tf
                self.keras_model = tf.keras.models.load_model(self.keras_path)
                logger.info(f"Loaded Keras Tier-2 Stacked CNN from {self.keras_path}")
            except Exception as e:
                logger.warning(f"Failed to load Keras model from {self.keras_path}: {e}")
        else:
            logger.warning(f"Keras model path not found: {self.keras_path}")

    def preprocess_keras(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Standardizes input for Keras CNN (224x224 RGB with channel mean subtraction)."""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB').resize((224, 224))
        elif isinstance(image, Image.Image):
            img = image.convert('RGB').resize((224, 224))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img = Image.fromarray(image).convert('RGB').resize((224, 224))
            elif image.shape[:2] != (224, 224):
                img = Image.fromarray(image.astype(np.uint8)).resize((224, 224))
            else:
                img = Image.fromarray(image.astype(np.uint8))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        arr = (np.array(img, dtype=np.float32) - CHANNEL_MEANS) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str],
        weights: Tuple[float, float, float] = DEFAULT_BRANCH_WEIGHTS,
        method: str = "weighted_average",
    ) -> Dict[str, Any]:
        """
        Runs full tri-branch inference on a single image.

        Returns:
            Dict containing:
                - 'predicted_class': Top predicted class name.
                - 'confidence': Highest confidence percentage.
                - 'fused_probabilities': (8,) numpy array.
                - 'fused_probabilities_dict': {class: prob} mapping.
                - 'branch_probabilities': {
                    'swin': np.ndarray,
                    'keras': np.ndarray,
                    'descriptor': np.ndarray,
                  }
                - 'visual_metrics': Dict of interpretable XAI clinical biomarkers.
        """
        # 1. Swin Transformer branch
        swin_probs = self.swin_classifier.predict_proba(image)

        # 2. Keras Tier-2 Stacked CNN branch
        if self.keras_model is not None:
            keras_tensor = self.preprocess_keras(image)
            keras_probs = self.keras_model.predict(keras_tensor, verbose=0)[0].astype(np.float32)
        else:
            keras_probs = swin_probs.copy()

        # 3. Handcrafted Descriptor branch
        if self.descriptor_classifier.is_loaded():
            desc_probs = self.descriptor_classifier.predict_proba(image)
        else:
            desc_probs = swin_probs.copy()

        # 4. Probability Fusion
        fused_probs = fuse_tri_branch_probabilities(
            swin_probs=swin_probs,
            keras_probs=keras_probs,
            desc_probs=desc_probs,
            method=method,
            weights=weights,
            meta_learner=self.meta_learner,
        )

        # 5. Extract Interpretable Visual Metrics (XAI)
        visual_metrics = self.descriptor_classifier.extract_interpretable_metrics(image)

        top_idx = int(np.argmax(fused_probs))
        top_class = self.classes[top_idx]
        confidence = float(fused_probs[top_idx] * 100.0)

        return {
            "predicted_class": top_class,
            "confidence": confidence,
            "fused_probabilities": fused_probs,
            "fused_probabilities_dict": {c: float(p) for c, p in zip(self.classes, fused_probs)},
            "branch_probabilities": {
                "swin": swin_probs,
                "keras": keras_probs,
                "descriptor": desc_probs,
            },
            "visual_metrics": visual_metrics,
        }


def train_stacking_meta_learner(
    train_dir: str,
    output_path: str = DEFAULT_META_LEARNER_PATH,
    n_splits: int = 5,
    random_state: int = 42,
) -> None:
    """
    Executes Out-of-Fold cross-validation to train the Tier-3 Stacking Meta-Learner
    on concatenated branch probabilities [P_swin, P_keras, P_desc] in R^24 -> R^8.
    """
    logger.info("Initializing TriBranchMetaLearner for OOF stacking training...")
    orchestrator = TriBranchMetaLearner()

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')
    image_paths: List[str] = []
    labels: List[int] = []

    for cls_idx, cls_name in enumerate(CLASSES):
        folder = os.path.join(train_dir, cls_name)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(valid_exts):
                image_paths.append(os.path.join(folder, f))
                labels.append(cls_idx)

    N = len(image_paths)
    y = np.array(labels, dtype=np.int64)
    logger.info(f"Loaded {N} training samples across 8 classes.")

    # Generate or load base model probabilities
    logger.info("Extracting branch probabilities for stacking...")
    X_stacked = np.zeros((N, 24), dtype=np.float32)

    for i, path in enumerate(image_paths):
        res = orchestrator.predict(path)
        swin_p = res["branch_probabilities"]["swin"]
        keras_p = res["branch_probabilities"]["keras"]
        desc_p = res["branch_probabilities"]["descriptor"]
        X_stacked[i] = np.concatenate([swin_p, keras_p, desc_p])

    # Out-of-fold cross validation for meta-learner
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_meta_probs = np.zeros((N, len(CLASSES)), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_stacked, y)):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state + fold)
        clf.fit(X_stacked[train_idx], y[train_idx])
        oof_meta_probs[val_idx] = clf.predict_proba(X_stacked[val_idx])

    oof_acc = accuracy_score(y, np.argmax(oof_meta_probs, axis=1))
    logger.info(f"OOF Stacking Meta-Learner Cross-Validation Accuracy: {oof_acc:.2%}")

    # Final fit on full dataset
    final_meta = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    final_meta.fit(X_stacked, y)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(final_meta, output_path)
    logger.info(f"Saved Stacking Meta-Learner to {output_path}")
