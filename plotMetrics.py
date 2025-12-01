import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image


# Path to the trained model and dataset
MODEL_PATH = 'BrainTumorModel.h5'
DATASET_DIR = 'datasets'      # Folder containing "no" and "yes" subfolders
INPUT_SIZE = 64
CLASSES = ['No Tumor', 'Yes Tumor']


def load_images(base_dir, input_size):
    """
    Load all images from the dataset folder structure:

        datasets/
          no/
          yes/

    Each image is resized to the given input_size and normalized to [0, 1].
    Returns:
        X: np.array of images
        y: np.array of labels (0 = no, 1 = yes)
    """
    X, y = [], []
    for label_name, label in [('no', 0), ('yes', 1)]:
        folder = os.path.join(base_dir, label_name)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(folder, fname)

            img = cv2.imread(path)
            if img is None:
                continue

            # OpenCV → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize and normalize
            img = Image.fromarray(img).resize((input_size, input_size))
            X.append(np.array(img, dtype=np.float32) / 255.0)
            y.append(label)
    return np.array(X), np.array(y)


# Load model and dataset
print("Loading model and dataset...")
model = load_model(MODEL_PATH, compile=False)
X, y = load_images(DATASET_DIR, INPUT_SIZE)

# Use 20% of the data as a test split
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Get prediction probabilities from the model
y_prob = model.predict(X_test, verbose=0)

# Handle both binary-sigmoid and 2-class softmax outputs
if y_prob.shape[1] == 1:
    y_score = y_prob.ravel()
    y_pred = (y_score >= 0.5).astype(int)
else:
    y_score = y_prob[:, 1]
    y_pred = np.argmax(y_prob, axis=1)


# If you saved any training metrics/curves earlier, show them here
if os.path.exists("metrics_artifacts/metrics_summary.json"):
    with open("metrics_artifacts/metrics_summary.json", "r") as f:
        data = json.load(f)
    print("\nLoaded metrics summary:")
    for k, v in data.items():
        print(f"{k}: {v}")

if os.path.exists("metrics_artifacts/accuracy_curve.png"):
    img = plt.imread("metrics_artifacts/accuracy_curve.png")
    plt.figure("Training Accuracy Curve")
    plt.imshow(img)
    plt.axis('off')
    plt.title("Training Accuracy Curve")
    plt.show()

if os.path.exists("metrics_artifacts/loss_curve.png"):
    img = plt.imread("metrics_artifacts/loss_curve.png")
    plt.figure("Training Loss Curve")
    plt.imshow(img)
    plt.axis('off')
    plt.title("Training Loss Curve")
    plt.show()


# Confusion matrix: how many were correctly / incorrectly classified
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(
            j, i,
            cm[i, j],
            ha='center',
            va='center',
            color='white' if cm[i, j] > cm.max() / 2 else 'black',
            fontsize=12
        )

plt.xticks([0, 1], CLASSES)
plt.yticks([0, 1], CLASSES)
plt.colorbar()
plt.show()


# ROC curve: trade-off between TPR and FPR
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Precision–Recall curve: useful for imbalanced data
precision, recall, _ = precision_recall_curve(y_test, y_score)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
