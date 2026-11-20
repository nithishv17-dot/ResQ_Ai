import os
import sys

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

# 1. Classes in exact alphabetical order matching model output layer
CLASSES = [
    'ants',
    'bed_bugs',
    'chiggers',
    'fleas',
    'mosquitos',
    'no_bites',
    'spiders',
    'ticks'
]

# Supported image file extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')

# 2. Dataset Channel Means (R, G, B) computed during training
CHANNEL_MEANS = np.array([170.88408, 134.76701, 118.84133], dtype=np.float32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'bug_bite_8class_stacked_model.keras')

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.\n")

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocesses input image with 224x224 RGB resize and channel mean subtraction."""
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    # Channel-wise mean subtraction as per training pipeline
    normalized = (img_array - CHANNEL_MEANS) / 255.0
    return np.expand_dims(normalized, axis=0)

def predict(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print("=" * 50)
    print(f"IMAGE           : {os.path.basename(image_path)}")
    print(f"PATH            : {image_path}")
    print("=" * 50)

    input_tensor = preprocess_image(image_path)
    probabilities = model.predict(input_tensor, verbose=0)[0]
    
    top_index = int(np.argmax(probabilities))
    top_class = CLASSES[top_index]
    confidence = probabilities[top_index] * 100

    print(f"PREDICTED CLASS : {top_class.upper()}")
    print(f"CONFIDENCE      : {confidence:.2f}%")
    print("-" * 50)
    print("Class Breakdown:")
    
    # Sort and display descending class probabilities
    ranked = sorted(zip(CLASSES, probabilities), key=lambda x: x[1], reverse=True)
    for class_name, prob in ranked:
        bar = "#" * int(prob * 20)
        print(f"  {class_name:<12}: {prob * 100:6.2f}%  {bar}")
    print("=" * 50 + "\n")

def get_images_from_dir(directory: str) -> list[str]:
    """Returns all image filepaths in the given directory."""
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(VALID_EXTENSIONS)
    ]

if __name__ == '__main__':
    # 1. Check if an image or folder path is passed via command line
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isdir(target):
            images = get_images_from_dir(target)
            if not images:
                print(f"No valid images found in directory '{target}'.")
            for img in images:
                predict(img)
        elif os.path.isfile(target):
            predict(target)
        else:
            print(f"Error: '{target}' is not a valid file or directory.")
    else:
        # 2. Automatically find and test any image(s) in the 'test_images' folder
        test_dir = os.path.join(BASE_DIR, 'test_images')
        images = get_images_from_dir(test_dir)
        
        if not images:
            print(f"No test images found in '{test_dir}/'.")
            print("Usage Options:")
            print("  1. Drop your test images into the 'test_images/' folder and run: python predict.py")
            print("  2. Or pass an image path directly: python predict.py path/to/image.jpg")
        else:
            print(f"Found {len(images)} image(s) in '{test_dir}/':")
            for img in images:
                predict(img)