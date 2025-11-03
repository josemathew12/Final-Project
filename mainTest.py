import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- Load model ---
model = load_model('BrainTumorModel.h5', compile=False)

# --- Read and preprocess image ---
image_path = '/Users/josemathew/Documents/Final Project/archive/pred/pred5.jpg'

# Read with OpenCV
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not read image at: {image_path}")

# Convert BGR → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PIL for resizing
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]

# Expand dimensions → (1, 64, 64, 3)
img = np.expand_dims(img, axis=0)

# --- Predict ---
prediction = model.predict(img)[0]

# --- Handle depending on model type ---
if len(prediction) == 1:
    # Binary model (sigmoid)
    prob_yes = float(prediction[0])
    label = "YES (Tumor)" if prob_yes >= 0.5 else "NO (Normal)"
    confidence = prob_yes if label.startswith("YES") else 1 - prob_yes
else:
    # Softmax model (2 outputs)
    classes = ['NO (Normal)', 'YES (Tumor)']
    label_idx = int(np.argmax(prediction))
    label = classes[label_idx]
    confidence = float(prediction[label_idx])

# --- Print result ---
print(f"Prediction: {label}")
print(f"Confidence: {confidence*100:.2f}%")
