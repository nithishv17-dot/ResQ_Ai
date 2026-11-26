"""
src/models/descriptor_model.py
==============================
Trained Handcrafted Visual Descriptor Base Classifier for ResQ_Ai Phase 2.

Implements:
1. scikit-learn Pipeline:
   - StandardScaler: Feature normalization across 583 handcrafted dimensions.
   - PCA(n_components=0.95, random_state=42): Captures 95% total variance.
   - CalibratedClassifierCV(estimator=LogisticRegression(...), cv=5, method='sigmoid'):
     Calibrated 8-class probability outputs.
2. DescriptorClassifier:
   - Inference adapter for single image feature extraction & calibrated prediction.
   - Persistence via joblib to `models/descriptor_classifier.joblib`.
3. Training Pipeline:
   - `train_descriptor_classifier`: Extracts features across local training splits
     and fits the calibrated pipeline.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.features.image_descriptors import BugBiteDescriptorExtractor

logger = logging.getLogger("DescriptorClassifier")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "descriptor_classifier.joblib")

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


class DescriptorClassifier:
    """
    Inference adapter for the Handcrafted Visual Descriptor Pipeline.
    Loads `models/descriptor_classifier.joblib` and evaluates single images.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        extractor: Optional[BugBiteDescriptorExtractor] = None,
        classes: List[str] = CLASSES,
    ):
        self.model_path = model_path or os.getenv("DESCRIPTOR_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.classes = list(classes)
        self.extractor = extractor or BugBiteDescriptorExtractor()
        self.pipeline: Optional[Pipeline] = None

        if os.path.exists(self.model_path):
            self.load(self.model_path)
        else:
            logger.warning(
                f"Descriptor classifier checkpoint not found at: {self.model_path}. "
                "Pipeline must be trained or loaded before running inference."
            )

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self, model_path: str) -> None:
        """Loads fitted pipeline from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            self.pipeline = loaded["pipeline"]
            if "classes" in loaded:
                self.classes = loaded["classes"]
        else:
            self.pipeline = loaded
        logger.info(f"Loaded DescriptorClassifier pipeline from: {model_path}")

    def save(self, output_path: str) -> None:
        """Saves fitted pipeline and metadata to disk."""
        if self.pipeline is None:
            raise ValueError("No pipeline to save.")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "classes": self.classes,
        }
        joblib.dump(payload, output_path)
        logger.info(f"Saved DescriptorClassifier pipeline to: {output_path}")

    def predict_proba(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Extracts visual descriptors and computes calibrated 8-class probabilities.

        Args:
            image: Image array, PIL Image, or file path.

        Returns:
            np.ndarray: Calibrated probability distribution of shape (8,) summing to 1.0.
        """
        if self.pipeline is None:
            raise RuntimeError("DescriptorClassifier pipeline is not loaded. Train or load first.")

        features = self.extractor.extract_features(image).reshape(1, -1)
        probs = self.pipeline.predict_proba(features)[0].astype(np.float32)

        # Ensure simplex sum == 1.0
        s = np.sum(probs)
        if s > 0:
            probs /= s

        return probs

    def predict_proba_detailed(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, float]:
        """Returns class-to-probability mapping."""
        probs = self.predict_proba(image)
        return {c_name: float(p) for c_name, p in zip(self.classes, probs)}

    def extract_interpretable_metrics(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, float]:
        """Extracts XAI visual and clinical biomarkers for display."""
        return self.extractor.extract_interpretable_metrics(image)


def train_descriptor_classifier(
    data_dir: str,
    output_path: str = DEFAULT_MODEL_PATH,
    classes: List[str] = CLASSES,
    random_state: int = 42,
    cv_folds: int = 5,
) -> DescriptorClassifier:
    """
    Extracts features from the training dataset split and fits the calibrated pipeline:
    StandardScaler -> PCA(n_components=0.95) -> CalibratedClassifierCV(LogisticRegression).

    Args:
        data_dir: Root directory containing class subfolders (e.g. 'training/').
        output_path: Destination path for saving descriptor_classifier.joblib.
        classes: List of 8 target classes.
        random_state: Reproducibility seed.
        cv_folds: Number of calibration folds.

    Returns:
        DescriptorClassifier: Trained and persisted classifier instance.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Training data directory does not exist: {data_dir}")

    extractor = BugBiteDescriptorExtractor()
    feature_rows = []
    labels = []

    logger.info(f"Extracting handcrafted descriptors from training directory: {data_dir}")

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')

    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            logger.warning(f"Class folder missing: {class_folder}")
            continue

        filenames = [f for f in os.listdir(class_folder) if f.endswith(valid_exts)]
        logger.info(f"Processing '{class_name}' ({len(filenames)} images)...")

        for fname in filenames:
            img_path = os.path.join(class_folder, fname)
            try:
                feat = extractor.extract_features(img_path)
                feature_rows.append(feat)
                labels.append(class_idx)
            except Exception as e:
                logger.error(f"Error extracting features from {img_path}: {e}")

    X = np.vstack(feature_rows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    logger.info(f"Extracted feature matrix shape: {X.shape}, labels: {y.shape}")

    # Build Pipeline: StandardScaler -> PCA(0.95) -> CalibratedClassifierCV
    base_lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=random_state,
        solver='lbfgs',
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=random_state)),
        ('classifier', CalibratedClassifierCV(estimator=base_lr, cv=cv_folds, method='sigmoid')),
    ])

    logger.info("Fitting Pipeline (StandardScaler -> PCA(0.95) -> CalibratedClassifierCV)...")
    pipeline.fit(X, y)
    logger.info("Pipeline training complete.")

    pca_step = pipeline.named_steps['pca']
    logger.info(
        f"PCA retained {pca_step.n_components_} components "
        f"(Explained variance ratio sum: {np.sum(pca_step.explained_variance_ratio_):.4f})"
    )

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "classes": classes,
        "n_samples": len(y),
        "n_components": int(pca_step.n_components_),
    }
    joblib.dump(payload, output_path)
    logger.info(f"Saved trained descriptor classifier checkpoint to: {output_path}")

    classifier_instance = DescriptorClassifier(
        model_path=output_path,
        extractor=extractor,
        classes=classes,
    )
    return classifier_instance


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Handcrafted Visual Descriptor Classifier.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\Career\Dataset\.dataset\datasets\external\stage1\insect-bite-dataset\training",
        help="Path to training image directory with 8 class subfolders.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Output path for descriptor_classifier.joblib.",
    )
    args = parser.parse_args()

    train_descriptor_classifier(args.data_dir, args.output_path)
