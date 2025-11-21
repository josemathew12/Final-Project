import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = 'BrainTumorModel.h5'
INPUT_SIZE = 64

# --- Load model ---
model = load_model(MODEL_PATH, compile=False)

# --- Read and preprocess image ---
image_path = '/Users/josemathew/Documents/Final Project/archive/pred/pred5.jpg'

image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not read image at: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = Image.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE))
img = np.array(img, dtype=np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# --- Predict ---
prediction = model.predict(img, verbose=0)[0]

if len(prediction) == 1:
    prob_yes = float(prediction[0])
    label = "YES (Tumor)" if prob_yes >= 0.5 else "NO (Normal)"
    confidence = prob_yes if label.startswith("YES") else 1 - prob_yes
else:
    classes = ['NO (Normal)', 'YES (Tumor)']
    label_idx = int(np.argmax(prediction))
    label = classes[label_idx]
    confidence = float(prediction[label_idx])

print(f"Prediction: {label}")
print(f"Confidence: {confidence*100:.2f}%")
