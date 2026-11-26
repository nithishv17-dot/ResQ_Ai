"""
test_tri_branch_pipeline.py
===========================
Comprehensive Test Suite for BugBiteDescriptorExtractor, DescriptorClassifier,
TriBranchMetaLearner, and FastAPI REST Backend.
"""

import os
import sys
import pathlib
import numpy as np
import pytest
from PIL import Image

BASE_DIR = pathlib.Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.features.image_descriptors import BugBiteDescriptorExtractor
from src.models.descriptor_model import DescriptorClassifier, DEFAULT_MODEL_PATH, CLASSES
from src.models.meta_learner import (
    TriBranchMetaLearner,
    fuse_tri_branch_probabilities,
    DEFAULT_BRANCH_WEIGHTS,
)
from api import app
from fastapi.testclient import TestClient

TEST_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__),
    "test_images",
    "es_spidersimage207 (copy).jpg",
)


# ---------------------------------------------------------------------------
# 1. BugBiteDescriptorExtractor Tests
# ---------------------------------------------------------------------------

def test_extractor_feature_vector_shape_and_finite():
    """Validates extractor output shape (583,), float32 dtype, and absence of NaNs/Infs."""
    extractor = BugBiteDescriptorExtractor()

    # 1. Synthetic image
    synth_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    feat = extractor.extract_features(synth_rgb)

    assert isinstance(feat, np.ndarray)
    assert feat.shape == (583,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))
    assert np.isnan(feat).sum() == 0
    assert np.isinf(feat).sum() == 0

    # 2. Real image
    if os.path.exists(TEST_IMAGE_PATH):
        feat_real = extractor.extract_features(TEST_IMAGE_PATH)
        assert feat_real.shape == (583,)
        assert np.all(np.isfinite(feat_real))


def test_extractor_determinism():
    """Asserts that consecutive runs on the exact same image produce bit-exact vectors."""
    extractor = BugBiteDescriptorExtractor()
    synth_rgb = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

    feat_1 = extractor.extract_features(synth_rgb)
    feat_2 = extractor.extract_features(synth_rgb)

    np.testing.assert_array_equal(feat_1, feat_2)


def test_extractor_input_types():
    """Validates support for NumPy arrays (uint8, float), PIL Images, and file paths."""
    extractor = BugBiteDescriptorExtractor()

    # PIL Image
    pil_img = Image.new("RGB", (300, 300), color=(200, 100, 50))
    feat_pil = extractor.extract_features(pil_img)
    assert feat_pil.shape == (583,)

    # Float array in [0, 1]
    float_arr = np.random.rand(224, 224, 3).astype(np.float32)
    feat_float = extractor.extract_features(float_arr)
    assert feat_float.shape == (583,)

    # 2D Grayscale array
    gray_arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    feat_gray = extractor.extract_features(gray_arr)
    assert feat_gray.shape == (583,)


def test_extractor_interpretable_metrics():
    """Validates XAI clinical biomarkers dictionary structure and numerical bounds."""
    extractor = BugBiteDescriptorExtractor()
    synth_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    metrics = extractor.extract_interpretable_metrics(synth_rgb)
    required_keys = {
        "erythema_score",
        "lesion_circularity",
        "texture_contrast",
        "lesion_area_ratio",
        "necrotic_center_contrast",
        "lesion_solidity",
        "lesion_eccentricity",
    }
    assert required_keys.issubset(metrics.keys())

    assert 0.0 <= metrics["erythema_score"] <= 1.0
    assert 0.0 <= metrics["lesion_circularity"] <= 1.0
    assert 0.0 <= metrics["lesion_solidity"] <= 1.0
    assert 0.0 <= metrics["lesion_eccentricity"] <= 1.0
    assert metrics["lesion_area_ratio"] >= 0.0


# ---------------------------------------------------------------------------
# 2. DescriptorClassifier Tests
# ---------------------------------------------------------------------------

def test_descriptor_classifier_prediction():
    """Validates checkpoint loading, predict_proba simplex sum, and class mappings."""
    classifier = DescriptorClassifier(model_path=DEFAULT_MODEL_PATH)
    assert classifier.is_loaded()

    if os.path.exists(TEST_IMAGE_PATH):
        probs = classifier.predict_proba(TEST_IMAGE_PATH)
        assert probs.shape == (8,)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

        prob_dict = classifier.predict_proba_detailed(TEST_IMAGE_PATH)
        assert len(prob_dict) == 8
        assert set(prob_dict.keys()) == set(CLASSES)


# ---------------------------------------------------------------------------
# 3. TriBranchMetaLearner & Probability Fusion Tests
# ---------------------------------------------------------------------------

def test_fuse_tri_branch_probabilities():
    """Validates weighted average probability fusion mathematics."""
    p_swin = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01], dtype=np.float32)
    p_keras = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02], dtype=np.float32)
    p_desc = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02], dtype=np.float32)

    fused = fuse_tri_branch_probabilities(
        swin_probs=p_swin,
        keras_probs=p_keras,
        desc_probs=p_desc,
        weights=(0.45, 0.40, 0.15),
    )

    assert fused.shape == (8,)
    assert np.isclose(np.sum(fused), 1.0, atol=1e-5)
    # Class 0 should be top
    assert np.argmax(fused) == 0
    # Expected: 0.45*0.7 + 0.40*0.5 + 0.15*0.4 = 0.315 + 0.20 + 0.06 = 0.575
    assert np.isclose(fused[0], 0.575, atol=1e-3)


def test_tri_branch_meta_learner_inference():
    """Validates full end-to-end tri-branch inference with visual XAI biomarkers."""
    learner = TriBranchMetaLearner()

    if os.path.exists(TEST_IMAGE_PATH):
        res = learner.predict(TEST_IMAGE_PATH)

        assert "predicted_class" in res
        assert res["predicted_class"] in CLASSES
        assert 0.0 <= res["confidence"] <= 100.0

        assert res["fused_probabilities"].shape == (8,)
        assert np.isclose(np.sum(res["fused_probabilities"]), 1.0, atol=1e-5)

        branches = res["branch_probabilities"]
        assert "swin" in branches
        assert "keras" in branches
        assert "descriptor" in branches
        assert branches["swin"].shape == (8,)
        assert branches["keras"].shape == (8,)
        assert branches["descriptor"].shape == (8,)

        assert "visual_metrics" in res
        assert "erythema_score" in res["visual_metrics"]
        assert "lesion_circularity" in res["visual_metrics"]


# ---------------------------------------------------------------------------
# 4. FastAPI REST Backend Tests
# ---------------------------------------------------------------------------

def test_fastapi_health():
    """Validates GET /health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert data["models"]["descriptor_classifier"] is True
    assert len(data["classes"]) == 8


def test_fastapi_classes():
    """Validates GET /classes endpoint."""
    client = TestClient(app)
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["classes"] == CLASSES
    assert data["total"] == 8


def test_fastapi_predict_endpoint():
    """Validates POST /predict endpoint with actual image upload."""
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image not found at {TEST_IMAGE_PATH}")

    client = TestClient(app)
    with open(TEST_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": ("test_spider.jpg", file_bytes, "image/jpeg")},
        data={"mode": "tri_branch"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in CLASSES
    assert "confidence" in data
    assert "fused_probabilities" in data
    assert len(data["fused_probabilities"]) == 8
    assert "branch_probabilities" in data
    assert "visual_metrics_xai" in data
    assert "erythema_score" in data["visual_metrics_xai"]
    assert "lesion_circularity" in data["visual_metrics_xai"]
