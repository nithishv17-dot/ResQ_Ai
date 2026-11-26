"""
test_swin_integration.py
========================
Test suite for PyTorch Swin Transformer integration (swin_phase2_best.pth)
and dual-vision ensemble fusion with Keras Tier-2 Stacked CNN.
"""

import os
import unittest
import numpy as np
from PIL import Image
import torch

from swin_classifier import (
    SwinClassifier,
    load_swin_model,
    fuse_vision_probabilities,
    train_transform,
    val_transform,
    inference_transform,
    CLASSES,
    SWIN_MODEL_PATH,
)


class TestSwinModelLoading(unittest.TestCase):
    """Verifies PyTorch Swin-Base architecture and checkpoint loading."""

    def test_checkpoint_exists(self):
        self.assertTrue(
            os.path.exists(SWIN_MODEL_PATH),
            f"Checkpoint file missing at: {SWIN_MODEL_PATH}",
        )

    def test_model_loading_and_eval_mode(self):
        model, metadata = load_swin_model(SWIN_MODEL_PATH)
        self.assertFalse(model.training, "Model should be in eval() mode")
        self.assertEqual(metadata.get("model_name"), "swin_base_patch4_window7_224")
        self.assertEqual(metadata.get("class_names"), CLASSES)
        self.assertAlmostEqual(metadata.get("val_acc", 0), 0.815165, places=4)

    def test_forward_pass_logits_and_softmax(self):
        classifier = SwinClassifier(SWIN_MODEL_PATH)
        dummy_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        top_c, conf, probs = classifier.predict(dummy_img)

        self.assertIn(top_c, CLASSES)
        self.assertEqual(probs.shape, (8,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 100.0)


class TestSwinTransforms(unittest.TestCase):
    """Verifies data augmentations and deterministic inference transforms."""

    def test_train_transform_shape_and_range(self):
        img = Image.new("RGB", (300, 300), color=(50, 100, 150))
        tensor = train_transform(img)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_inference_transform_deterministic(self):
        img = Image.new("RGB", (256, 256), color=(200, 100, 50))
        tensor1 = inference_transform(img)
        tensor2 = inference_transform(img)
        self.assertTrue(torch.allclose(tensor1, tensor2))
        self.assertEqual(tensor1.shape, (3, 224, 224))


class TestEnsembleFusion(unittest.TestCase):
    """Verifies probability averaging and modality calibration."""

    def test_equal_weight_fusion(self):
        keras_p = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
        swin_p = np.array([0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)

        fused = fuse_vision_probabilities(keras_p, swin_p, method="weighted_average", weights=(0.5, 0.5))
        self.assertEqual(len(fused), 8)
        self.assertAlmostEqual(float(np.sum(fused)), 1.0, places=5)
        self.assertAlmostEqual(fused[0], 0.25, places=5)
        self.assertAlmostEqual(fused[1], 0.25, places=5)

    def test_real_image_inference(self):
        test_img_path = os.path.join("test_images", "es_spidersimage207 (copy).jpg")
        if os.path.exists(test_img_path):
            classifier = SwinClassifier(SWIN_MODEL_PATH)
            top_c, conf, probs = classifier.predict(test_img_path)
            self.assertEqual(top_c, "spiders")
            self.assertGreater(conf, 50.0)


if __name__ == "__main__":
    unittest.main()
