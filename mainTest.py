import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Path to the trained model file
MODEL_PATH = 'BrainTumorModel.h5'

# Input image size expected by the model (64x64 for your CNN)
INPUT_SIZE = 64

# Load the trained model once
model = load_model(MODEL_PATH, compile=False)

# Path to the MRI image you want to test
image_path = '/Users/josemathew/Documents/Final Project/archive/pred/pred5.jpg'

# Read the image using OpenCV
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not read image at: {image_path}")

# Convert BGR (OpenCV default) to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PIL, resize to model input size
img = Image.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE))

# Convert to float32 numpy array and normalize to [0, 1]
img = np.array(img, dtype=np.float32) / 255.0

# Add batch dimension → shape becomes (1, 64, 64, 3)
img = np.expand_dims(img, axis=0)

# Run the model prediction
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
