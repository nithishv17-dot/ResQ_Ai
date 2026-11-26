import os
import sys
import argparse
import pathlib

# Suppress TensorFlow / oneDNN debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Safe UTF-8 encoding configuration for Windows consoles
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv

from download_model import ensure_model
from swin_classifier import SwinClassifier, fuse_vision_probabilities, CLASSES
from src.models.descriptor_model import DescriptorClassifier
from src.models.meta_learner import fuse_tri_branch_probabilities, DEFAULT_BRANCH_WEIGHTS

load_dotenv()

# Supported image file extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')

# Dataset Channel Means (R, G, B) for Keras models
CHANNEL_MEANS = np.array([170.88408, 134.76701, 118.84133], dtype=np.float32)

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_KERAS_PATH = str(BASE_DIR / "models" / "bug_bite_8class_stacked_model.keras")
DEFAULT_SWIN_PATH = str(BASE_DIR / "models" / "swin_phase2_best.pth")
DEFAULT_DESCRIPTOR_PATH = str(BASE_DIR / "models" / "descriptor_classifier.joblib")

KERAS_MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_KERAS_PATH)
SWIN_MODEL_PATH = os.getenv("SWIN_MODEL_PATH", DEFAULT_SWIN_PATH)
DESCRIPTOR_MODEL_PATH = os.getenv("DESCRIPTOR_MODEL_PATH", DEFAULT_DESCRIPTOR_PATH)

print("=" * 68)
print("Initializing Vision Classifiers (Keras Stacked + Swin-B + Descriptors)...")
print("=" * 68)

# 1. Load Keras Tier-2 Stacking Model
ensure_model(KERAS_MODEL_PATH)
keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
print("✅ Keras Tier-2 Stacking Ensemble loaded.")

# 2. Load PyTorch Swin-Base Model
swin_classifier = SwinClassifier(model_path=SWIN_MODEL_PATH)
print(f"✅ PyTorch Swin Transformer loaded (Val Acc: {swin_classifier.metadata.get('val_acc', 0.8152):.2%}).")

# 3. Load Handcrafted Visual Descriptor Model
descriptor_classifier = DescriptorClassifier(model_path=DESCRIPTOR_MODEL_PATH)
print("✅ Handcrafted Visual Descriptor Classifier loaded.\n")


def preprocess_keras_image(image_path: str) -> np.ndarray:
    """Preprocesses input image with 224x224 RGB resize and channel mean subtraction."""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    normalized = (img_array - CHANNEL_MEANS) / 255.0
    return np.expand_dims(normalized, axis=0)


def predict(image_path: str, mode: str = "tri_branch"):
    """
    Performs multi-branch inference on a single image.

    Args:
        image_path: Path to target image.
        mode: 'tri_branch' (fused 3 models), 'ensemble' (Swin + CNN),
              'swin' (PyTorch only), 'keras' (CNN only), or 'descriptor' (handcrafted).
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("=" * 75)
    print(f"IMAGE : {os.path.basename(image_path)}")
    print(f"PATH  : {image_path}")
    print(f"MODE  : {mode.upper()}")
    print("=" * 75)

    # 1. Keras CNN Prediction
    keras_tensor = preprocess_keras_image(image_path)
    keras_probs = keras_model.predict(keras_tensor, verbose=0)[0]

    # 2. PyTorch Swin Prediction
    swin_probs = swin_classifier.predict_proba(image_path)

    # 3. Handcrafted Visual Descriptor Prediction
    desc_probs = descriptor_classifier.predict_proba(image_path)

    # 4. Fused Predictions
    # Dual Vision Ensemble (50% Swin + 50% CNN)
    dual_probs = fuse_vision_probabilities(
        keras_probs, swin_probs, method="weighted_average", weights=(0.50, 0.50)
    )
    # Tri-Branch Multimodal Ensemble (45% Swin + 40% CNN + 15% Descriptors)
    tri_probs = fuse_tri_branch_probabilities(
        swin_probs, keras_probs, desc_probs, method="weighted_average", weights=DEFAULT_BRANCH_WEIGHTS
    )

    if mode == "swin":
        active_probs = swin_probs
        model_label = "PyTorch Swin-B Transformer"
    elif mode == "keras":
        active_probs = keras_probs
        model_label = "Keras Tier-2 Stacked CNN"
    elif mode == "descriptor":
        active_probs = desc_probs
        model_label = "Handcrafted Visual Descriptors (Color/Texture/SIFT/Morphology)"
    elif mode == "ensemble":
        active_probs = dual_probs
        model_label = "Dual Vision Ensemble (Swin-B + Stacked CNN)"
    else:  # tri_branch
        active_probs = tri_probs
        model_label = "Tri-Branch Multimodal Ensemble (Swin-B + CNN + Descriptors)"

    top_index = int(np.argmax(active_probs))
    top_class = CLASSES[top_index]
    confidence = float(active_probs[top_index] * 100)

    # 5. Extract Explainable AI (XAI) Biomarkers
    metrics = descriptor_classifier.extract_interpretable_metrics(image_path)

    print(f"ACTIVE MODEL    : {model_label}")
    print(f"PREDICTED CLASS : {top_class.upper()}")
    print(f"CONFIDENCE      : {confidence:.2f}%")
    print("-" * 75)
    print(f"{'Class':<12} | {'Tri-Branch':>10} | {'Swin-B':>8} | {'Stacked':>8} | {'Descriptor':>10}")
    print("-" * 75)

    ranked_indices = np.argsort(active_probs)[::-1]
    for idx in ranked_indices:
        c_name = CLASSES[idx]
        tri_p = tri_probs[idx] * 100
        swin_p = swin_probs[idx] * 100
        keras_p = keras_probs[idx] * 100
        desc_p = desc_probs[idx] * 100
        bar = "#" * int(active_probs[idx] * 15)
        print(f"{c_name:<12} | {tri_p:9.2f}% | {swin_p:7.2f}% | {keras_p:7.2f}% | {desc_p:9.2f}%  {bar}")

    print("-" * 75)
    print("🔬 INTERPRETABLE CLINICAL & VISUAL BIOMARKERS (XAI):")
    print(f"  • Erythema (Redness) Score   : {metrics['erythema_score']:.4f} (LAB a* elevation)")
    print(f"  • Lesion Circularity         : {metrics['lesion_circularity']:.4f} (1.0 = circular target rash)")
    print(f"  • Necrotic Center Contrast   : {metrics['necrotic_center_contrast']:.4f} (L* center darkness depression)")
    print(f"  • Texture Contrast           : {metrics['texture_contrast']:.4f} (GLCM micro-roughness)")
    print(f"  • Lesion Area Coverage       : {metrics['lesion_area_ratio'] * 100:.2f}% of image field")
    print(f"  • Lesion Solidity            : {metrics['lesion_solidity']:.4f}")
    print(f"  • Lesion Eccentricity        : {metrics['lesion_eccentricity']:.4f}")
    print("=" * 75 + "\n")

    return top_class, confidence, active_probs, metrics


def get_images_from_dir(directory: str) -> list[str]:
    """Returns all valid image filepaths in the given directory."""
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(VALID_EXTENSIONS)
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multimodal Insect Bite Vision Classifier")
    parser.add_argument("target", nargs="?", default=None, help="Image path or folder")
    parser.add_argument(
        "--mode",
        choices=["tri_branch", "ensemble", "swin", "keras", "descriptor"],
        default="tri_branch",
        help="Inference mode: 'tri_branch' (default), 'ensemble', 'swin', 'keras', or 'descriptor'",
    )

    args = parser.parse_args()

    if args.target:
        if os.path.isdir(args.target):
            images = get_images_from_dir(args.target)
            if not images:
                print(f"No valid images found in directory '{args.target}'.")
            for img in images:
                predict(img, mode=args.mode)
        elif os.path.isfile(args.target):
            predict(args.target, mode=args.mode)
        else:
            print(f"Error: '{args.target}' is not a valid file or directory.")
    else:
        test_dir = os.path.join(str(BASE_DIR), 'test_images')
        images = get_images_from_dir(test_dir)

        if not images:
            print(f"No test images found in '{test_dir}/'.")
        else:
            for img in images:
                predict(img, mode=args.mode)
