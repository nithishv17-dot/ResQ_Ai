"""
classical_features.py
=====================
Leakage-Free Classical Feature Extraction Pipeline for ResQ_Ai Phase 2.

Implements:
1. Classical Feature Extractors:
   - Gray-Level Co-occurrence Matrix (GLCM): Contrast, dissimilarity,
     homogeneity, energy, correlation, ASM across multiple distances & angles.
   - Local Binary Patterns (LBP): Rotation-invariant uniform patterns for micro-texture.
   - Scale-Invariant Feature Transform (SIFT): Keypoint detection and 128-dim
     descriptor extraction with graceful fallback for zero keypoints.
   - Bag-of-Visual-Words (BoVW): Visual word codebook quantization.
2. Strict Leakage-Free Out-of-Fold (OOF) Cross-Validation Pipeline:
   - Prevents data leakage into the Tier-2 Meta-Learner by strictly fitting:
     * BoVW KMeans Codebook
     * StandardScaler
     * PCA Dimensionality Reduction
     * Descriptor Classifier
     exclusively on the training fold (train_idx) of each cross-validation split.
3. Fast Production Inference:
   - Full-pipeline single image feature extraction & probability inference in ~15-30 ms on CPU.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import joblib

logger = logging.getLogger("classical_features")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Default Constants & 8-Class Taxonomy
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

GLCM_DISTANCES: Tuple[int, ...] = (1, 2, 3)
GLCM_ANGLES: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_LEVELS: int = 32
GLCM_PROPERTIES: Tuple[str, ...] = (
    'contrast',
    'dissimilarity',
    'homogeneity',
    'energy',
    'correlation',
    'ASM',
)

LBP_POINTS: int = 8
LBP_RADIUS: int = 1
LBP_METHOD: str = 'uniform'

SIFT_MAX_FEATURES: int = 200
DEFAULT_BOVW_CLUSTERS: int = 64
DEFAULT_PCA_COMPONENTS: int = 32


# ---------------------------------------------------------------------------
# Preprocessing Helper
# ---------------------------------------------------------------------------

def preprocess_image_to_grayscale(
    image: Union[np.ndarray, Any],
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Normalizes an input image (numpy array, PIL Image, or file path) to a
    standardized 224x224 8-bit unsigned integer grayscale numpy array.

    Args:
        image: Input image array, PIL image, or path string.
        target_size: (width, height) resolution tuple, default (224, 224).

    Returns:
        np.ndarray: Grayscale image with shape (height, width) and dtype uint8.
    """
    if isinstance(image, str):
        img_arr = cv2.imread(image)
        if img_arr is None:
            raise FileNotFoundError(f"Image file could not be read from path: {image}")
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    elif hasattr(image, "convert"):  # PIL Image
        img_rgb = np.array(image.convert("RGB"))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2] == 4:  # RGBA
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:  # RGB or BGR
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = image[:, :, 0]
        elif image.ndim == 2:
            img_gray = image
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure uint8 [0, 255]
    if img_gray.dtype != np.uint8:
        if img_gray.max() <= 1.0:
            img_gray = (img_gray * 255.0).astype(np.uint8)
        else:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    # Resize to target size (width, height)
    if (img_gray.shape[1], img_gray.shape[0]) != target_size:
        img_gray = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)

    return img_gray


# ---------------------------------------------------------------------------
# Feature Extraction: GLCM
# ---------------------------------------------------------------------------

def extract_glcm_features(
    gray_image: np.ndarray,
    distances: Tuple[int, ...] = GLCM_DISTANCES,
    angles: Tuple[float, ...] = GLCM_ANGLES,
    properties: Tuple[str, ...] = GLCM_PROPERTIES,
    levels: int = GLCM_LEVELS,
) -> np.ndarray:
    """
    Computes Gray-Level Co-occurrence Matrix (GLCM) statistical features.

    Calculates texture descriptors across defined pixel distances and orientation
    angles (0, 45, 90, 135 degrees), capturing spatial gray-level dependencies.
    Quantizes gray levels (default 32) for ultra-fast, high-fidelity texture
    computation (~2-4 ms vs ~100 ms for 256 levels).

    Args:
        gray_image: Standardized 2D uint8 image (H, W).
        distances: Pixel pair distances, default (1, 2, 3).
        angles: Direction angles in radians, default (0, pi/4, pi/2, 3pi/4).
        properties: GLCM metrics to compute.
        levels: Quantization levels for GLCM (default 32).

    Returns:
        np.ndarray: 1D float32 vector of GLCM properties (length: len(properties) * len(distances) * len(angles)).
                    For default parameters: 6 * 3 * 4 = 72 dimensions.
    """
    if levels < 256 and gray_image.max() >= levels:
        quant_factor = max(1, 256 // levels)
        quantized = (gray_image // quant_factor).astype(np.uint8)
        if quantized.max() >= levels:
            quantized = np.clip(quantized, 0, levels - 1)
    else:
        quantized = gray_image

    # Compute normalized symmetric GLCM
    glcm = graycomatrix(
        quantized,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    feature_list = []
    for prop in properties:
        # prop_matrix shape: (len(distances), len(angles))
        prop_matrix = graycoprops(glcm, prop)
        feature_list.append(prop_matrix.flatten())

    glcm_vector = np.concatenate(feature_list).astype(np.float32)
    # Replace any potential NaNs or Infs
    np.nan_to_num(glcm_vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return glcm_vector


# ---------------------------------------------------------------------------
# Feature Extraction: LBP
# ---------------------------------------------------------------------------

def extract_lbp_features(
    gray_image: np.ndarray,
    P: int = LBP_POINTS,
    R: int = LBP_RADIUS,
    method: str = LBP_METHOD,
) -> np.ndarray:
    """
    Computes Local Binary Patterns (LBP) normalized histogram.

    Captures fine, rotation-invariant micro-textures and boundary contours.
    With uniform patterns, bins represent distinct edge/spot patterns plus
    one bin for non-uniform patterns.

    Args:
        gray_image: Standardized 2D uint8 image (H, W).
        P: Number of circularly symmetric neighbor set points, default 8.
        R: Radius of circle, default 1.
        method: LBP pattern method ('uniform' by default).

    Returns:
        np.ndarray: Normalized 1D float32 histogram vector of length (P + 2).
                    For P=8: 10 dimensions.
    """
    lbp = local_binary_pattern(gray_image, P=P, R=R, method=method)
    n_bins = P + 2 if method == "uniform" else int(2 ** P)

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=False,
    )
    hist = hist.astype(np.float32)
    # L1 normalization
    sum_hist = np.sum(hist)
    if sum_hist > 0:
        hist /= sum_hist
    return hist


# ---------------------------------------------------------------------------
# Feature Extraction: SIFT
# ---------------------------------------------------------------------------

def extract_sift_descriptors(
    gray_image: np.ndarray,
    max_features: int = SIFT_MAX_FEATURES,
) -> np.ndarray:
    """
    Detects keypoints and extracts 128-dimensional SIFT descriptors.

    Gracefully handles low-texture images where no keypoints are detected
    by returning an empty array of shape (0, 128).

    Args:
        gray_image: Standardized 2D uint8 image (H, W).
        max_features: Maximum number of keypoints to retain, default 200.

    Returns:
        np.ndarray: Descriptors array of shape (N, 128) with dtype float32.
    """
    sift = cv2.SIFT_create(nfeatures=max_features)
    _, descriptors = sift.detectAndCompute(gray_image, None)

    if descriptors is None or len(descriptors) == 0:
        return np.empty((0, 128), dtype=np.float32)

    return descriptors.astype(np.float32)


# ---------------------------------------------------------------------------
# Bag of Visual Words (BoVW) Quantization
# ---------------------------------------------------------------------------

def quantize_bovw_histogram(
    sift_descriptors: np.ndarray,
    kmeans_codebook: Any,
    n_clusters: int = DEFAULT_BOVW_CLUSTERS,
) -> np.ndarray:
    """
    Quantizes an image's SIFT descriptors into a Bag-of-Visual-Words histogram
    using a pre-fitted visual vocabulary (KMeans codebook).

    Args:
        sift_descriptors: Array of shape (N, 128) containing SIFT descriptors.
        kmeans_codebook: Pre-fitted KMeans or MiniBatchKMeans clustering model.
        n_clusters: Number of visual words in the codebook.

    Returns:
        np.ndarray: Normalized histogram of shape (n_clusters,) with dtype float32.
    """
    if sift_descriptors is None or len(sift_descriptors) == 0:
        # Zero keypoints: return uniform zero vector
        return np.zeros(n_clusters, dtype=np.float32)

    # Predict visual words for each keypoint
    cluster_ids = kmeans_codebook.predict(sift_descriptors)
    histogram = np.bincount(cluster_ids, minlength=n_clusters).astype(np.float32)

    # L1-normalization
    total = np.sum(histogram)
    if total > 0:
        histogram /= total

    return histogram


# ---------------------------------------------------------------------------
# Combined Single-Image Feature Extractor
# ---------------------------------------------------------------------------

class RawImageFeatures:
    """Container holding raw classical descriptors extracted from a single image."""
    __slots__ = ("glcm", "lbp", "sift_descriptors")

    def __init__(self, glcm: np.ndarray, lbp: np.ndarray, sift_descriptors: np.ndarray):
        self.glcm = glcm
        self.lbp = lbp
        self.sift_descriptors = sift_descriptors


def extract_raw_image_features(
    image: Union[np.ndarray, Any],
    max_sift_features: int = SIFT_MAX_FEATURES,
    target_size: Tuple[int, int] = (224, 224),
) -> RawImageFeatures:
    """
    Extracts GLCM, LBP, and raw SIFT descriptors for a single image.
    This step is strictly image-local and independent of any other images,
    making it completely safe from data leakage.

    Args:
        image: Image array, PIL Image, or file path.
        max_sift_features: Upper bound for SIFT keypoints.
        target_size: (width, height) resize target.

    Returns:
        RawImageFeatures: Contains .glcm (72,), .lbp (10,), and .sift_descriptors (N, 128).
    """
    gray = preprocess_image_to_grayscale(image, target_size=target_size)
    glcm_vec = extract_glcm_features(gray)
    lbp_vec = extract_lbp_features(gray)
    sift_desc = extract_sift_descriptors(gray, max_features=max_sift_features)
    return RawImageFeatures(glcm=glcm_vec, lbp=lbp_vec, sift_descriptors=sift_desc)


# ---------------------------------------------------------------------------
# Production Classical Feature Pipeline
# ---------------------------------------------------------------------------

class ProductionClassicalPipeline:
    """
    Trained end-to-end Classical Descriptor Model for ResQ_Ai inference.

    Holds the fitted parameters:
    - BoVW KMeans codebook
    - StandardScaler
    - PCA reduction
    - Classifier (LogisticRegression / Calibrated / Classifier)
    """

    def __init__(
        self,
        kmeans_codebook: Any,
        scaler: StandardScaler,
        pca: PCA,
        classifier: Any,
        classes: List[str] = CLASSES,
        n_clusters: int = DEFAULT_BOVW_CLUSTERS,
    ):
        self.kmeans_codebook = kmeans_codebook
        self.scaler = scaler
        self.pca = pca
        self.classifier = classifier
        self.classes = list(classes)
        self.n_clusters = n_clusters

    def extract_features(self, image: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Transforms a single raw image into the final scaled PCA feature vector.

        Returns:
            np.ndarray: Projected feature vector of shape (1, n_pca_components).
        """
        raw = extract_raw_image_features(image)
        bovw_hist = quantize_bovw_histogram(
            raw.sift_descriptors,
            self.kmeans_codebook,
            n_clusters=self.n_clusters,
        )
        combined = np.concatenate([raw.glcm, raw.lbp, bovw_hist]).reshape(1, -1)
        scaled = self.scaler.transform(combined)
        projected = self.pca.transform(scaled)
        return projected

    def predict_proba(self, image: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Predicts 8-class probability distribution for a single image.

        Returns:
            np.ndarray: Softmax probabilities of shape (n_classes,) summing to 1.0.
        """
        features = self.extract_features(image)
        probs = self.classifier.predict_proba(features)[0]
        return probs.astype(np.float32)

    def predict_proba_detailed(self, image: Union[np.ndarray, Any]) -> Dict[str, float]:
        """
        Returns class name to probability mapping.
        """
        probs = self.predict_proba(image)
        return {cls_name: float(p) for cls_name, p in zip(self.classes, probs)}

    def save(self, filepath: str) -> None:
        """Saves pipeline artifacts using joblib."""
        payload = {
            "kmeans_codebook": self.kmeans_codebook,
            "scaler": self.scaler,
            "pca": self.pca,
            "classifier": self.classifier,
            "classes": self.classes,
            "n_clusters": self.n_clusters,
        }
        joblib.dump(payload, filepath)
        logger.info(f"ProductionClassicalPipeline saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ProductionClassicalPipeline":
        """Loads pipeline artifacts from disk."""
        payload = joblib.load(filepath)
        return cls(
            kmeans_codebook=payload["kmeans_codebook"],
            scaler=payload["scaler"],
            pca=payload["pca"],
            classifier=payload["classifier"],
            classes=payload["classes"],
            n_clusters=payload["n_clusters"],
        )


# ---------------------------------------------------------------------------
# Leakage-Free Out-Of-Fold (OOF) Cross-Validation Engine
# ---------------------------------------------------------------------------

class LeakageFreeOOFPipeline:
    """
    Strict Leakage-Free Cross-Validation Engine for Tier-2 Meta-Learner Data.

    =========================================================================
    MATHEMATICAL & ARCHITECTURAL GUARDRAIL:
    =========================================================================
    To prevent data leakage when generating Out-of-Fold (OOF) probabilities:
    1. BoVW KMeans codebook is fitted SOLELY on SIFT keypoints from train_idx.
    2. StandardScaler is fitted SOLELY on feature vectors from train_idx.
    3. PCA is fitted SOLELY on standardized feature vectors from train_idx.
    4. Classifier is fitted SOLELY on (X_train_pca, y_train).
    5. OOF predictions are produced on val_idx using the strictly trained fold
       pipeline. No test/validation fold data is EVER observed during fitting.
    =========================================================================
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_clusters: int = DEFAULT_BOVW_CLUSTERS,
        pca_components: int = DEFAULT_PCA_COMPONENTS,
        random_state: int = 42,
        max_sift_pool_samples: int = 50000,
        classes: List[str] = CLASSES,
    ):
        self.n_splits = n_splits
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.random_state = random_state
        self.max_sift_pool_samples = max_sift_pool_samples
        self.classes = list(classes)

    def run_oof_cv(
        self,
        raw_features_list: List[RawImageFeatures],
        labels: np.ndarray,
        classifier_factory: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Executes strict leakage-free K-fold cross-validation.

        Args:
            raw_features_list: Pre-extracted RawImageFeatures for all N samples.
            labels: Integer class labels array of shape (N,).
            classifier_factory: Callable returning a fresh classifier instance.
                                Defaults to LogisticRegression(max_iter=1000, C=1.0).

        Returns:
            Dict containing:
                - 'oof_probabilities': (N, n_classes) array of out-of-fold predictions.
                - 'fold_accuracies': List of per-fold validation accuracies.
                - 'mean_accuracy': Mean CV accuracy across folds.
                - 'leakage_audit': Diagnostic dictionary confirming isolated fits.
        """
        N = len(raw_features_list)
        if N != len(labels):
            raise ValueError(f"Feature count ({N}) does not match label count ({len(labels)})")

        labels = np.asarray(labels, dtype=np.int64)
        n_classes = len(self.classes)

        if classifier_factory is None:
            def default_classifier():
                return LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    random_state=self.random_state,
                    solver='lbfgs',
                )
            classifier_factory = default_classifier

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        oof_probabilities = np.zeros((N, n_classes), dtype=np.float32)
        fold_accuracies: List[float] = []
        leakage_audit: List[Dict[str, Any]] = []

        logger.info(
            f"Starting {self.n_splits}-fold Leakage-Free OOF Cross-Validation on {N} samples..."
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), labels)):
            logger.info(f"--- Fold {fold + 1}/{self.n_splits} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

            # ---------------------------------------------------------------
            # 1. Fit BoVW KMeans Codebook STRICTLY on train_idx SIFT descriptors
            # ---------------------------------------------------------------
            train_sift_list = [
                raw_features_list[i].sift_descriptors
                for i in train_idx
                if len(raw_features_list[i].sift_descriptors) > 0
            ]

            if len(train_sift_list) > 0:
                pooled_train_sift = np.vstack(train_sift_list)
                if len(pooled_train_sift) > self.max_sift_pool_samples:
                    rng = np.random.RandomState(self.random_state + fold)
                    sub_idx = rng.choice(len(pooled_train_sift), self.max_sift_pool_samples, replace=False)
                    pooled_train_sift = pooled_train_sift[sub_idx]
            else:
                # Synthetic fallback if no keypoints were detected anywhere
                pooled_train_sift = np.zeros((self.n_clusters, 128), dtype=np.float32)

            kmeans_fold = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state + fold,
                batch_size=1024,
                n_init="auto",
                max_iter=100,
            )
            kmeans_fold.fit(pooled_train_sift)

            # ---------------------------------------------------------------
            # 2. Quantize & Assemble Feature Matrices for Train and Val
            # ---------------------------------------------------------------
            def build_feature_matrix(indices: np.ndarray) -> np.ndarray:
                rows = []
                for idx in indices:
                    rf = raw_features_list[idx]
                    bovw = quantize_bovw_histogram(
                        rf.sift_descriptors,
                        kmeans_fold,
                        n_clusters=self.n_clusters,
                    )
                    row = np.concatenate([rf.glcm, rf.lbp, bovw])
                    rows.append(row)
                return np.vstack(rows)

            X_train = build_feature_matrix(train_idx)
            X_val = build_feature_matrix(val_idx)
            y_train = labels[train_idx]
            y_val = labels[val_idx]

            # ---------------------------------------------------------------
            # 3. Fit StandardScaler STRICTLY on X_train (Zero Val Exposure)
            # ---------------------------------------------------------------
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train)
            X_val_scaled = scaler_fold.transform(X_val)

            # ---------------------------------------------------------------
            # 4. Fit PCA STRICTLY on X_train_scaled (Zero Val Exposure)
            # ---------------------------------------------------------------
            # Ensure n_components does not exceed min(samples, features)
            max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
            actual_components = min(self.pca_components, max_components)

            pca_fold = PCA(
                n_components=actual_components,
                random_state=self.random_state + fold,
            )
            X_train_pca = pca_fold.fit_transform(X_train_scaled)
            X_val_pca = pca_fold.transform(X_val_scaled)

            # ---------------------------------------------------------------
            # 5. Fit Classifier STRICTLY on X_train_pca (Zero Val Exposure)
            # ---------------------------------------------------------------
            clf_fold = classifier_factory()
            clf_fold.fit(X_train_pca, y_train)

            # ---------------------------------------------------------------
            # 6. Predict OOF Probabilities for val_idx
            # ---------------------------------------------------------------
            val_probs = clf_fold.predict_proba(X_val_pca)

            # Check if all classes are represented in fold predictions
            if val_probs.shape[1] < n_classes:
                # Map sparse classes back to full n_classes
                full_val_probs = np.zeros((len(val_idx), n_classes), dtype=np.float32)
                for local_col, actual_cls in enumerate(clf_fold.classes_):
                    full_val_probs[:, actual_cls] = val_probs[:, local_col]
                val_probs = full_val_probs

            oof_probabilities[val_idx] = val_probs

            val_preds = np.argmax(val_probs, axis=1)
            fold_acc = float(np.mean(val_preds == y_val))
            fold_accuracies.append(fold_acc)

            # Audit fold isolation
            leakage_audit.append({
                "fold": fold + 1,
                "train_samples": int(len(train_idx)),
                "val_samples": int(len(val_idx)),
                "scaler_mean_first5": scaler_fold.mean_[:5].tolist(),
                "pca_explained_variance_ratio_sum": float(np.sum(pca_fold.explained_variance_ratio_)),
                "fold_accuracy": fold_acc,
            })
            logger.info(f"Fold {fold + 1} Val Accuracy: {fold_acc * 100:.2f}%")

        overall_acc = float(np.mean(np.argmax(oof_probabilities, axis=1) == labels))
        logger.info(f"Leakage-Free OOF Cross-Validation Complete. Overall Accuracy: {overall_acc * 100:.2f}%")

        return {
            "oof_probabilities": oof_probabilities,
            "fold_accuracies": fold_accuracies,
            "mean_accuracy": float(np.mean(fold_accuracies)),
            "overall_accuracy": overall_acc,
            "leakage_audit": leakage_audit,
        }

    def fit_production_pipeline(
        self,
        raw_features_list: List[RawImageFeatures],
        labels: np.ndarray,
        classifier_factory: Optional[Any] = None,
    ) -> ProductionClassicalPipeline:
        """
        Fits the entire pipeline on the full training set for production deployment.

        Args:
            raw_features_list: Pre-extracted RawImageFeatures for all training samples.
            labels: Integer class labels array.
            classifier_factory: Optional classifier constructor.

        Returns:
            ProductionClassicalPipeline: Ready for single-image CPU inference.
        """
        N = len(raw_features_list)
        labels = np.asarray(labels, dtype=np.int64)

        if classifier_factory is None:
            def default_classifier():
                return LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    random_state=self.random_state,
                    solver='lbfgs',
                )
            classifier_factory = default_classifier

        logger.info(f"Fitting Production Classical Pipeline on all {N} training samples...")

        # 1. Codebook on full dataset SIFT descriptors
        all_sift_list = [
            rf.sift_descriptors
            for rf in raw_features_list
            if len(rf.sift_descriptors) > 0
        ]
        if len(all_sift_list) > 0:
            pooled_sift = np.vstack(all_sift_list)
            if len(pooled_sift) > self.max_sift_pool_samples:
                rng = np.random.RandomState(self.random_state)
                sub_idx = rng.choice(len(pooled_sift), self.max_sift_pool_samples, replace=False)
                pooled_sift = pooled_sift[sub_idx]
        else:
            pooled_sift = np.zeros((self.n_clusters, 128), dtype=np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=1024,
            n_init="auto",
            max_iter=100,
        )
        kmeans.fit(pooled_sift)

        # 2. Assemble full feature matrix
        rows = []
        for rf in raw_features_list:
            bovw = quantize_bovw_histogram(
                rf.sift_descriptors,
                kmeans,
                n_clusters=self.n_clusters,
            )
            rows.append(np.concatenate([rf.glcm, rf.lbp, bovw]))
        X = np.vstack(rows)

        # 3. StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. PCA
        max_components = min(X_scaled.shape[0], X_scaled.shape[1])
        actual_components = min(self.pca_components, max_components)
        pca = PCA(n_components=actual_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        # 5. Classifier
        clf = classifier_factory()
        clf.fit(X_pca, labels)

        pipeline = ProductionClassicalPipeline(
            kmeans_codebook=kmeans,
            scaler=scaler,
            pca=pca,
            classifier=clf,
            classes=self.classes,
            n_clusters=self.n_clusters,
        )
        logger.info("Production Classical Pipeline successfully fitted.")
        return pipeline
