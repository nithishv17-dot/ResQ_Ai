import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Path to your local testing directory
TEST_DIR = r"D:\Career\Dataset\.dataset\datasets\external\stage1\insect-bite-dataset\testing"
MODEL_PATH = os.path.join(BASE_DIR, "models", "bug_bite_8class_stacked_model.keras")

# 2. Classes matching the model's output layer
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

# 3. Channel means from training pipeline
CHANNEL_MEANS = np.array([170.88408, 134.76701, 118.84133], dtype=np.float32)

def paper_preprocess(x):
    return (x - CHANNEL_MEANS) / 255.0

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.\n")

# 4. Create evaluation data generator
eval_datagen = ImageDataGenerator(preprocessing_function=paper_preprocess)

test_gen = eval_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    classes=CLASSES,
    class_mode='categorical',
    shuffle=False
)

# 5. Run batch evaluation
print("\n--- Running Evaluation on Local Test Split ---")
loss, acc = model.evaluate(test_gen, verbose=1)
print("=" * 50)
print(f"Total Local Test Samples : {test_gen.samples}")
print(f"Overall Test Accuracy    : {acc * 100:.2f}%")
print(f"Overall Test Loss        : {loss:.4f}")
print("=" * 50)

# 6. Detailed per-class precision / recall breakdown
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

print("\n--- Per-Class Performance Breakdown ---")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))