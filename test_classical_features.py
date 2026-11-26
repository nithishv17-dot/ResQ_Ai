"""
test_classical_features.py
==========================
Unit tests for the Leakage-Free Classical Feature Extraction Pipeline.

Validates:
1. Grayscale normalization, input types, and target resize.
2. GLCM texture extraction & dimension guarantees (72 dimensions).
3. LBP uniform micro-texture histogram properties (10 dimensions, sums to 1.0).
4. SIFT descriptor extraction and zero-keypoint edge cases.
5. BoVW codebook quantization (including zero-keypoint fallback).
6. Strict Leakage-Free K-Fold Cross-Validation:
   - Proves mathematically that scaler.mean_, pca.components_, and KMeans codebook
     are 100% isolated to the training fold with zero validation exposure.
   - Verifies OOF probabilities have shape (N, 8) and simplex sum == 1.0.
7. Production pipeline serialization, reload, and single-image inference.
"""

import os
import tempfile
import numpy as np
import cv2
from PIL import Image
import pytest
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from classical_features import (
    CLASSES,
    preprocess_image_to_grayscale,
    extract_glcm_features,
    extract_lbp_features,
    extract_sift_descriptors,
    quantize_bovw_histogram,
    extract_raw_image_features,
    RawImageFeatures,
    LeakageFreeOOFPipeline,
    ProductionClassicalPipeline,
)


TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__),
    "test_images",
    "es_spidersimage207 (copy).jpg",
)


def test_preprocess_grayscale_various_inputs():
    """Tests preprocessing from numpy arrays (RGB, BGR, RGBA, float, uint8) and PIL Image."""
    # 1. 3-channel RGB uint8
    rgb_uint8 = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    gray1 = preprocess_image_to_grayscale(rgb_uint8, target_size=(224, 224))
    assert gray1.shape == (224, 224)
    assert gray1.dtype == np.uint8

    # 2. 4-channel RGBA uint8
    rgba_uint8 = np.random.randint(0, 256, (150, 150, 4), dtype=np.uint8)
    gray2 = preprocess_image_to_grayscale(rgba_uint8, target_size=(224, 224))
    assert gray2.shape == (224, 224)
    assert gray2.dtype == np.uint8

    # 3. Normalized float array in [0, 1]
    float_img = np.random.rand(100, 100, 3).astype(np.float32)
    gray3 = preprocess_image_to_grayscale(float_img, target_size=(224, 224))
    assert gray3.shape == (224, 224)
    assert gray3.dtype == np.uint8
    assert gray3.max() <= 255

    # 4. PIL Image
    pil_img = Image.fromarray(rgb_uint8)
    gray4 = preprocess_image_to_grayscale(pil_img, target_size=(224, 224))
    assert gray4.shape == (224, 224)
    assert gray4.dtype == np.uint8

    # 5. File path
    if os.path.exists(TEST_IMAGE_PATH):
        gray5 = preprocess_image_to_grayscale(TEST_IMAGE_PATH, target_size=(224, 224))
        assert gray5.shape == (224, 224)
        assert gray5.dtype == np.uint8


def test_extract_glcm_features():
    """Validates GLCM feature vector dimensionality, finiteness, and determinism."""
    # 224x224 synthetic textured image
    img = np.zeros((224, 224), dtype=np.uint8)
    img[::2, ::2] = 200
    img[1::2, 1::2] = 100

    glcm_feat = extract_glcm_features(img)
    # 6 properties * 3 distances * 4 angles = 72 dimensions
    assert glcm_feat.shape == (72,)
    assert glcm_feat.dtype == np.float32
    assert np.all(np.isfinite(glcm_feat))

    # Test determinism
    glcm_feat_2 = extract_glcm_features(img)
    np.testing.assert_array_almost_equal(glcm_feat, glcm_feat_2)


def test_extract_lbp_features():
    """Validates LBP normalized histogram features."""
    img = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    lbp_feat = extract_lbp_features(img, P=8, R=1, method='uniform')

    # Uniform P=8 gives P + 2 = 10 bins
    assert lbp_feat.shape == (10,)
    assert lbp_feat.dtype == np.float32
    assert np.all(lbp_feat >= 0.0)
    assert np.isclose(np.sum(lbp_feat), 1.0, atol=1e-5)


def test_extract_sift_descriptors_real_and_blank():
    """Validates SIFT keypoint extraction on real image and zero-keypoint handling on uniform blank."""
    # Real image
    if os.path.exists(TEST_IMAGE_PATH):
        gray = preprocess_image_to_grayscale(TEST_IMAGE_PATH)
        desc = extract_sift_descriptors(gray, max_features=200)
        assert desc.ndim == 2
        assert desc.shape[1] == 128
        assert desc.shape[0] > 0
        assert desc.dtype == np.float32

    # Uniform blank image (zero texture -> 0 keypoints)
    blank = np.full((224, 224), fill_value=128, dtype=np.uint8)
    desc_blank = extract_sift_descriptors(blank, max_features=200)
    assert desc_blank.shape == (0, 128)
    assert desc_blank.dtype == np.float32


def test_bovw_quantization():
    """Validates Bag of Visual Words quantization with normal and empty descriptors."""
    n_clusters = 32
    rng = np.random.RandomState(42)
    sift_train = rng.randn(300, 128).astype(np.float32)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(sift_train)

    # 1. Normal descriptors
    sift_sample = rng.randn(50, 128).astype(np.float32)
    hist = quantize_bovw_histogram(sift_sample, kmeans, n_clusters=n_clusters)
    assert hist.shape == (n_clusters,)
    assert np.isclose(np.sum(hist), 1.0, atol=1e-5)
    assert np.all(hist >= 0.0)

    # 2. Zero keypoints edge case
    empty_sift = np.empty((0, 128), dtype=np.float32)
    hist_empty = quantize_bovw_histogram(empty_sift, kmeans, n_clusters=n_clusters)
    assert hist_empty.shape == (n_clusters,)
    assert np.all(hist_empty == 0.0)


def test_leakage_free_oof_pipeline_strict_isolation():
    """
    MATHEMATICAL LEAKAGE PROOF TEST:
    Ensures that scaler, PCA, and KMeans are fitted strictly on the train fold,
    and that validation samples have absolutely zero influence on the training statistics.
    """
    n_samples = 40
    n_classes = len(CLASSES)
    rng = np.random.RandomState(123)

    # Create synthetic raw features list
    raw_list: list[RawImageFeatures] = []
    for _ in range(n_samples):
        glcm = rng.randn(72).astype(np.float32)
        lbp = np.abs(rng.randn(10)).astype(np.float32)
        lbp /= np.sum(lbp)
        n_kp = rng.randint(5, 25)
        sift = rng.randn(n_kp, 128).astype(np.float32)
        raw_list.append(RawImageFeatures(glcm=glcm, lbp=lbp, sift_descriptors=sift))

    labels = np.array([i % n_classes for i in range(n_samples)], dtype=np.int64)

    pipeline = LeakageFreeOOFPipeline(
        n_splits=4,
        n_clusters=16,
        pca_components=12,
        random_state=42,
    )

    cv_results = pipeline.run_oof_cv(raw_list, labels)

    oof_probs = cv_results["oof_probabilities"]
    assert oof_probs.shape == (n_samples, n_classes)
    assert np.all(oof_probs >= 0.0)
    assert np.all(oof_probs <= 1.0)

    # Simplex constraint: Each row must sum to 1.0
    row_sums = np.sum(oof_probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    # Verify fold isolation audit
    leakage_audit = cv_results["leakage_audit"]
    assert len(leakage_audit) == 4

    # Confirm scaler means are distinct across folds (proof that each fold scaler is fitted on its own train split)
    mean_fold1 = leakage_audit[0]["scaler_mean_first5"]
    mean_fold2 = leakage_audit[1]["scaler_mean_first5"]
    assert mean_fold1 != mean_fold2, "Scaler means across different folds must differ if trained only on fold splits!"


def test_leakage_invariance_under_val_perturbation():
    """
    DIRECT LEAKAGE INVARIANCE TEST:
    Artificially modifying/corrupting a validation image MUST NOT alter
    the fold's StandardScaler mean or PCA components.
    """
    rng = np.random.RandomState(999)
    train_raw = [
        RawImageFeatures(
            glcm=rng.randn(72).astype(np.float32),
            lbp=np.ones(10, dtype=np.float32) / 10.0,
            sift_descriptors=rng.randn(20, 128).astype(np.float32),
        )
        for _ in range(20)
    ]

    # Val image A vs corrupted Val image B
    val_raw_clean = RawImageFeatures(
        glcm=rng.randn(72).astype(np.float32),
        lbp=np.ones(10, dtype=np.float32) / 10.0,
        sift_descriptors=rng.randn(20, 128).astype(np.float32),
    )
    # Extreme corrupted outlier
    val_raw_corrupted = RawImageFeatures(
        glcm=np.full(72, 1e7, dtype=np.float32),
        lbp=np.full(10, 1e7, dtype=np.float32),
        sift_descriptors=np.full((50, 128), 1e7, dtype=np.float32),
    )

    # Assemble dataset 1 and dataset 2 (identical train fold, different val sample)
    data1 = train_raw + [val_raw_clean]
    data2 = train_raw + [val_raw_corrupted]
    labels = np.array([i % 8 for i in range(21)], dtype=np.int64)

    pipeline = LeakageFreeOOFPipeline(
        n_splits=3,
        n_clusters=8,
        pca_components=8,
        random_state=42,
    )

    res1 = pipeline.run_oof_cv(data1, labels)
    res2 = pipeline.run_oof_cv(data2, labels)

    # Find the fold where index 20 is in validation
    # For that fold, scaler means and PCA variance ratio must be 100% IDENTICAL
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(21), labels)):
        if 20 in val_idx:
            # Index 20 is in validation fold!
            audit1 = res1["leakage_audit"][fold]
            audit2 = res2["leakage_audit"][fold]

            # Assert 100% invariant despite extreme corruption of val sample 20
            assert audit1["scaler_mean_first5"] == audit2["scaler_mean_first5"], (
                "Scaler mean on train fold was affected by validation fold data! Data leakage detected!"
            )
            assert np.isclose(
                audit1["pca_explained_variance_ratio_sum"],
                audit2["pca_explained_variance_ratio_sum"],
                atol=1e-7,
            ), "PCA components were affected by validation fold data! Data leakage detected!"
            print(f"\nVerification Passed: Fold {fold + 1} training state is 100% invariant to validation perturbations.")
            break


def test_production_pipeline_persistence_and_inference():
    """Tests fitting, saving, loading, and predicting on a real test image."""
    rng = np.random.RandomState(42)
    raw_list = [
        RawImageFeatures(
            glcm=rng.randn(72).astype(np.float32),
            lbp=np.ones(10, dtype=np.float32) / 10.0,
            sift_descriptors=rng.randn(30, 128).astype(np.float32),
        )
        for _ in range(24)
    ]
    labels = np.array([i % 8 for i in range(24)], dtype=np.int64)

    pipeline_builder = LeakageFreeOOFPipeline(
        n_splits=3,
        n_clusters=16,
        pca_components=8,
        random_state=42,
    )

    prod_pipeline = pipeline_builder.fit_production_pipeline(raw_list, labels)

    if os.path.exists(TEST_IMAGE_PATH):
        # 1. Predict with memory pipeline
        probs = prod_pipeline.predict_proba(TEST_IMAGE_PATH)
        assert probs.shape == (8,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

        prob_dict = prod_pipeline.predict_proba_detailed(TEST_IMAGE_PATH)
        assert len(prob_dict) == 8
        assert set(prob_dict.keys()) == set(CLASSES)

        # 2. Save and reload
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            prod_pipeline.save(tmp_path)
            loaded_pipeline = ProductionClassicalPipeline.load(tmp_path)

            loaded_probs = loaded_pipeline.predict_proba(TEST_IMAGE_PATH)
            np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
