"""
evaluate_test_set.py
====================
Comprehensive Evaluation Pipeline for ResQ_Ai:
Evaluates Keras Tier-2 Stacking Ensemble, PyTorch Swin-B, and Fused Ensemble
on the local test dataset.
"""

import os
import sys
import pathlib
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from download_model import ensure_model
from swin_classifier import (
    SwinClassifier,
    fuse_vision_probabilities,
    CLASSES,
    SWIN_MODEL_PATH,
    BASE_DIR,
)
from src.models.descriptor_model import DescriptorClassifier
from src.models.meta_learner import fuse_tri_branch_probabilities, DEFAULT_BRANCH_WEIGHTS

load_dotenv()

# Path to local testing directory
DEFAULT_TEST_DIR = r"D:\Career\Dataset\.dataset\datasets\external\stage1\insect-bite-dataset\testing"
TEST_DIR = os.getenv("TEST_DIR", DEFAULT_TEST_DIR)

DEFAULT_KERAS_PATH = str(BASE_DIR / "models" / "bug_bite_8class_stacked_model.keras")
KERAS_MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_KERAS_PATH)

# Channel means from Keras training pipeline
CHANNEL_MEANS = np.array([170.88408, 134.76701, 118.84133], dtype=np.float32)


def paper_preprocess(x):
    """Channel-mean subtraction for Keras CNN."""
    return (x - CHANNEL_MEANS) / 255.0


def evaluate(test_dir: str = TEST_DIR, batch_size: int = 32):
    if not os.path.exists(test_dir):
        print(f"[Warning] Test dataset folder not found at: '{test_dir}'")
        print("Please set TEST_DIR environment variable or update the path.")
        return

    print("=" * 65)
    print("ResQ_Ai Test Split Dual-Architecture Evaluation")
    print(f"📁 Test Directory : {test_dir}")
    print(f"📦 Keras Model    : {KERAS_MODEL_PATH}")
    print(f"📦 PyTorch Swin   : {SWIN_MODEL_PATH}")
    print("=" * 65)

    # 1. Load Keras Model
    ensure_model(KERAS_MODEL_PATH)
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    print("✅ Loaded Keras Tier-2 Stacked CNN model.")

    # 2. Load PyTorch Swin Model
    swin_classifier = SwinClassifier(model_path=SWIN_MODEL_PATH)
    print(f"✅ Loaded PyTorch Swin-B model (Checkpoint Val Acc: {swin_classifier.metadata.get('val_acc', 0.8152):.2%}).")

    # 3. Load Handcrafted Visual Descriptor Model
    descriptor_classifier = DescriptorClassifier()
    print("✅ Loaded Handcrafted Visual Descriptor Classifier.\n")

    # 4. Create Keras ImageDataGenerator
    eval_datagen = ImageDataGenerator(preprocessing_function=paper_preprocess)
    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=CLASSES,
        class_mode='categorical',
        shuffle=False,
    )

    total_samples = test_gen.samples
    print(f"Total Test Samples: {total_samples}")
    y_true = test_gen.classes
    filenames = test_gen.filenames

    # 5. Keras Predictions
    print("\n[1/4] Generating Keras Stacked CNN predictions...")
    test_gen.reset()
    keras_probs = keras_model.predict(test_gen, verbose=1)

    # 6. PyTorch Swin Predictions
    print("\n[2/4] Generating PyTorch Swin-B predictions...")
    swin_probs_list = []
    for rel_path in filenames:
        full_path = os.path.join(test_dir, rel_path)
        p = swin_classifier.predict_proba(full_path)
        swin_probs_list.append(p)
    swin_probs = np.array(swin_probs_list, dtype=np.float32)

    # 7. Handcrafted Descriptor Predictions
    print("\n[3/4] Generating Handcrafted Descriptor predictions...")
    desc_probs_list = []
    for rel_path in filenames:
        full_path = os.path.join(test_dir, rel_path)
        p = descriptor_classifier.predict_proba(full_path)
        desc_probs_list.append(p)
    desc_probs = np.array(desc_probs_list, dtype=np.float32)

    # 8. Multi-Branch Fused Predictions
    print("\n[4/4] Computing Tri-Branch Multimodal Ensemble predictions (45% Swin + 40% Keras + 15% Descriptors)...")
    tri_probs = np.array([
        fuse_tri_branch_probabilities(s, k, d, weights=DEFAULT_BRANCH_WEIGHTS)
        for s, k, d in zip(swin_probs, keras_probs, desc_probs)
    ])

    # 9. Evaluate and Compare
    y_pred_keras = np.argmax(keras_probs, axis=1)
    y_pred_swin = np.argmax(swin_probs, axis=1)
    y_pred_desc = np.argmax(desc_probs, axis=1)
    y_pred_tri = np.argmax(tri_probs, axis=1)

    acc_keras = accuracy_score(y_true, y_pred_keras) * 100
    acc_swin = accuracy_score(y_true, y_pred_swin) * 100
    acc_desc = accuracy_score(y_true, y_pred_desc) * 100
    acc_tri = accuracy_score(y_true, y_pred_tri) * 100

    print("\n" + "=" * 68)
    print("📊 OVERALL ACCURACY SUMMARY ON TEST SPLIT")
    print("=" * 68)
    print(f"  PyTorch Swin-B Transformer       : {acc_swin:.2f}%")
    print(f"  Keras Stacked CNN Ensemble       : {acc_keras:.2f}%")
    print(f"  Handcrafted Visual Descriptors   : {acc_desc:.2f}%")
    print(f"  Tri-Branch Multimodal Ensemble   : {acc_tri:.2f}%")
    print("=" * 68)

    print("\n--- Tri-Branch Multimodal Ensemble Per-Class Classification Report ---")
    print(classification_report(y_true, y_pred_tri, target_names=CLASSES, digits=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Insect Bite Classifiers")
    parser.add_argument("--test_dir", default=TEST_DIR, help="Path to test image dataset")
    args = parser.parse_args()
    evaluate(test_dir=args.test_dir)
