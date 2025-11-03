import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import json

# -----------------------------
# Reproducibility + (optional) GPU memory growth
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# -----------------------------
# Parameters
# -----------------------------
image_directory = 'datasets/'   # your dataset folder path
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 20
CLASSES = ('no', 'yes')

# -----------------------------
# Dataset checking & loading
# -----------------------------
def check_dataset_dirs(base_dir: str, classes=CLASSES) -> None:
    for folder in classes:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created missing folder: {path}")
            print("Please add images (.jpg/.jpeg/.png) and re-run.")
            raise SystemExit(1)
        images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) == 0:
            print(f"Folder '{path}' exists but contains no images.")
            print("Please add images (.jpg/.jpeg/.png) and re-run.")
            raise SystemExit(1)
        print(f"Folder '{folder}' contains {len(images)} images")

def safe_read_resize(image_path: str, input_size: int):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
    else:
        try:
            pil = Image.open(image_path).convert('RGB')
        except UnidentifiedImageError:
            return None
    pil = pil.resize((input_size, input_size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0

def load_dataset(base_dir: str, input_size: int):
    dataset, labels, bad = [], [], 0
    for folder_name, class_label in [('no', 0), ('yes', 1)]:
        folder_path = os.path.join(base_dir, folder_name)
        for image_name in os.listdir(folder_path):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(folder_path, image_name)
            arr = safe_read_resize(image_path, input_size)
            if arr is None:
                bad += 1
                print(f"Warning: could not read {image_path}, skipping.")
                continue
            dataset.append(arr)
            labels.append(class_label)
    print(f"Skipped unreadable/corrupt files: {bad}")
    return np.array(dataset, dtype=np.float32), np.array(labels, dtype=np.int32)

# -----------------------------
# Main Training Process
# -----------------------------
if __name__ == '__main__':
    check_dataset_dirs(image_directory, classes=CLASSES)

    dataset, label = load_dataset(image_directory, INPUT_SIZE)
    print(f"Total images loaded: {len(dataset)}")
    if len(dataset) == 0:
        print("No images were loaded. Exiting.")
        raise SystemExit(1)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, label, test_size=0.2, random_state=SEED, shuffle=True, stratify=label
    )
    print("x_train shape:", x_train.shape)
    print("x_test shape :", x_test.shape)

    # One-hot encode
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Model definition
    model = Sequential()
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Training
    print("Starting training...")
    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True),
        validation_data=(x_test, y_test_cat),
        epochs=EPOCHS,
        verbose=1
    )

    # Save the model
    model.save('BrainTumorModel.h5')
    print("Model saved as BrainTumorModel.h5")

    # ===========================
    # Show Final Results
    # ===========================
    final_epoch = len(history.history['loss']) - 1
    train_acc  = history.history['accuracy'][final_epoch]
    val_acc    = history.history['val_accuracy'][final_epoch]
    train_loss = history.history['loss'][final_epoch]
    val_loss   = history.history['val_loss'][final_epoch]

    print("\n================= FINAL RESULTS =================")
    print(f"Training Accuracy:   {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Training Loss:       {train_loss:.4f}")
    print(f"Validation Loss:     {val_loss:.4f}")
    print("=================================================\n")

    # Save metrics to a file (optional)
    metrics = {
        "Training Accuracy": round(train_acc, 4),
        "Validation Accuracy": round(val_acc, 4),
        "Training Loss": round(train_loss, 4),
        "Validation Loss": round(val_loss, 4)
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to metrics.json")

    # Plot training history
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
