import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dense,
    Flatten,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Basic setup: random seeds and optional GPU memory growth
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# Training configuration
image_directory = 'datasets'   # folder that contains /no and /yes

# CNN model settings
INPUT_SIZE_CNN = 64
BATCH_SIZE_CNN = 16
EPOCHS_CNN = 25

# EfficientNetB0 settings
INPUT_SIZE_EFF = 224
BATCH_SIZE_EFF = 16
EPOCHS_EFF_FROZEN = 12   # stage 1: base frozen
EPOCHS_EFF_FT = 8        # stage 2: fine-tuning

CLASSES = ('no', 'yes')


# Dataset helpers
def check_dataset_dirs(base_dir: str, classes=CLASSES) -> None:
    """
    Make sure the dataset folders (e.g. datasets/no and datasets/yes)
    exist and contain at least one image each.
    """
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


def _read_image_generic(image_path: str):
    """
    Read an image from disk and return a PIL Image in RGB.
    Returns None if reading fails.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        # Handle grayscale, RGBA, and regular BGR images
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
    else:
        try:
            pil = Image.open(image_path).convert('RGB')
        except UnidentifiedImageError:
            return None
    return pil


def load_dataset_cnn(base_dir: str, input_size: int):
    """
    Load images for the CNN model.
    Images are resized and normalized to [0, 1].
    """
    dataset, labels, bad = [], [], 0
    for folder_name, class_label in [('no', 0), ('yes', 1)]:
        folder_path = os.path.join(base_dir, folder_name)
        for image_name in os.listdir(folder_path):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(folder_path, image_name)
            pil = _read_image_generic(image_path)
            if pil is None:
                bad += 1
                print(f"Warning: could not read {image_path}, skipping.")
                continue
            pil = pil.resize((input_size, input_size), Image.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            dataset.append(arr)
            labels.append(class_label)
    print(f"[CNN] Skipped unreadable/corrupt files: {bad}")
    return np.array(dataset, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_dataset_eff(base_dir: str, input_size: int):
    """
    Load images for the EfficientNet model.
    Images are resized then passed through EfficientNetB0 preprocess_input.
    """
    dataset, labels, bad = [], [], 0
    for folder_name, class_label in [('no', 0), ('yes', 1)]:
        folder_path = os.path.join(base_dir, folder_name)
        for image_name in os.listdir(folder_path):
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(folder_path, image_name)
            pil = _read_image_generic(image_path)
            if pil is None:
                bad += 1
                print(f"Warning: could not read {image_path}, skipping.")
                continue
            pil = pil.resize((input_size, input_size), Image.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32)
            arr = preprocess_input(arr)
            dataset.append(arr)
            labels.append(class_label)
    print(f"[EffNet] Skipped unreadable/corrupt files: {bad}")
    return np.array(dataset, dtype=np.float32), np.array(labels, dtype=np.int32)


# Model builders
def build_cnn(input_size):
    """
    Small CNN with three convolution blocks and a dense head
    for binary classification (no / yes).
    """
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(input_size, input_size, 3)))
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
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
    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model


def build_efficientnet(input_size):
    """
    Build an EfficientNetB0-based classifier:
    - use imagenet weights as the feature extractor
    - add a small dense head on top
    """
    base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base


# Utility to create output folders and plots
def ensure_artifacts_dir():
    """
    Create a folder train_artifacts/<timestamp> to store
    weights, plots, and metrics for this run.
    """
    out_root = Path("train_artifacts")
    out_root.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / ts
    run_dir.mkdir(exist_ok=True)
    return out_root, run_dir


def plot_curves(history, title_prefix, out_dir):
    """
    Save training and validation accuracy/loss curves
    as PNG images in the given output directory.
    """
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title_prefix} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_accuracy.png", dpi=160)
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{title_prefix.lower().replace(' ', '_')}_loss.png", dpi=160)
    plt.close()


# Main training entry point
if __name__ == '__main__':
    # Confirm dataset folders exist and contain images
    check_dataset_dirs(image_directory, classes=CLASSES)

    # Create a timestamped folder to store results
    artifacts_root, run_dir = ensure_artifacts_dir()

    #  MODEL 1: CNN (64x64)
    print("\nTRAINING MODEL 1: CNN (64x64)")
    dataset_cnn, label_cnn = load_dataset_cnn(image_directory, INPUT_SIZE_CNN)
    print(f"Total images loaded for CNN: {len(dataset_cnn)}")
    if len(dataset_cnn) == 0:
        print("No images were loaded. Exiting.")
        raise SystemExit(1)

    x_train1, x_val1, y_train1, y_val1 = train_test_split(
        dataset_cnn,
        label_cnn,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
        stratify=label_cnn
    )
    print("CNN x_train shape:", x_train1.shape)
    print("CNN x_val shape  :", x_val1.shape)

    y_train1_cat = to_categorical(y_train1, num_classes=2)
    y_val1_cat = to_categorical(y_val1, num_classes=2)

    # Basic augmentation for CNN training
    datagen_cnn = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen_cnn.fit(x_train1)

    model_cnn = build_cnn(INPUT_SIZE_CNN)
    model_cnn.summary()

    callbacks_cnn = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-5
        ),
        ModelCheckpoint(
            filepath=str(run_dir / "best_cnn.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    history_cnn = model_cnn.fit(
        datagen_cnn.flow(x_train1, y_train1_cat, batch_size=BATCH_SIZE_CNN, shuffle=True),
        validation_data=(x_val1, y_val1_cat),
        epochs=EPOCHS_CNN,
        callbacks=callbacks_cnn,
        verbose=1
    )

    # Reload the best CNN weights before final saving
    if (run_dir / "best_cnn.h5").exists():
        model_cnn.load_weights(run_dir / "best_cnn.h5")

    # Save CNN to two filenames:
    # - BrainTumorModel_CNN.h5 for experiments
    # - BrainTumorModel.h5 as the deployed model for the Flask app
    model_cnn.save('BrainTumorModel_CNN.h5')
    model_cnn.save('BrainTumorModel.h5')
    print("Model 1 saved as BrainTumorModel_CNN.h5 and BrainTumorModel.h5 (deployed model)")

    # Extract final CNN metrics
    final_epoch_cnn = len(history_cnn.history['loss']) - 1
    train_acc_cnn  = history_cnn.history['accuracy'][final_epoch_cnn]
    val_acc_cnn    = history_cnn.history['val_accuracy'][final_epoch_cnn]
    train_loss_cnn = history_cnn.history['loss'][final_epoch_cnn]
    val_loss_cnn   = history_cnn.history['val_loss'][final_epoch_cnn]

    print("\nMODEL 1 (CNN) FINAL RESULTS")
    print(f"Training Accuracy:   {train_acc_cnn*100:.2f}%")
    print(f"Validation Accuracy: {val_acc_cnn*100:.2f}%")
    print(f"Training Loss:       {train_loss_cnn:.4f}")
    print(f"Validation Loss:     {val_loss_cnn:.4f}")

    plot_curves(history_cnn, "CNN", run_dir)

    #  MODEL 2: EfficientNetB0 (224x224)
    print("\nTRAINING MODEL 2: EfficientNetB0 (224x224)")
    dataset_eff, label_eff = load_dataset_eff(image_directory, INPUT_SIZE_EFF)
    print(f"Total images loaded for EfficientNet: {len(dataset_eff)}")
    if len(dataset_eff) == 0:
        print("No images were loaded. Exiting.")
        raise SystemExit(1)

    x_train2, x_val2, y_train2, y_val2 = train_test_split(
        dataset_eff,
        label_eff,
        test_size=0.2,
        random_state=SEED,
        shuffle=True,
        stratify=label_eff
    )
    print("EfficientNet x_train shape:", x_train2.shape)
    print("EfficientNet x_val shape  :", x_val2.shape)

    y_train2_cat = to_categorical(y_train2, num_classes=2)
    y_val2_cat = to_categorical(y_val2, num_classes=2)

    # Augmentation for EfficientNet training
    datagen_eff = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True
    )
    datagen_eff.fit(x_train2)

    model_eff, base_eff = build_efficientnet(INPUT_SIZE_EFF)
    model_eff.summary()

    # Stage 1: train classifier head with the base frozen
    callbacks_eff_stage1 = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=str(run_dir / "best_eff_stage1.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]


    history_eff_stage1 = model_eff.fit(
        datagen_eff.flow(x_train2, y_train2_cat, batch_size=BATCH_SIZE_EFF, shuffle=True),
        validation_data=(x_val2, y_val2_cat),
        epochs=EPOCHS_EFF_FROZEN,
        callbacks=callbacks_eff_stage1,
        verbose=1
    )

    if (run_dir / "best_eff_stage1.h5").exists():
        model_eff.load_weights(run_dir / "best_eff_stage1.h5")

    # Stage 2: unfreeze top layers of EfficientNet and fine-tune
    print("\nUnfreezing top EfficientNet layers for fine-tuning...")
    for layer in base_eff.layers[:-40]:
        layer.trainable = False
    for layer in base_eff.layers[-40:]:
        layer.trainable = True

    model_eff.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_eff_stage2 = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=str(run_dir / "best_eff_overall.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    history_eff_stage2 = model_eff.fit(
        datagen_eff.flow(x_train2, y_train2_cat, batch_size=BATCH_SIZE_EFF, shuffle=True),
        validation_data=(x_val2, y_val2_cat),
        epochs=EPOCHS_EFF_FT,
        callbacks=callbacks_eff_stage2,
        verbose=1
    )

    # Combine stage 1 and stage 2 history for plotting and final metrics
    history_eff = history_eff_stage1
    for k in history_eff.history.keys():
        history_eff.history[k].extend(history_eff_stage2.history[k])

    if (run_dir / "best_eff_overall.h5").exists():
        model_eff.load_weights(run_dir / "best_eff_overall.h5")

    model_eff.save('BrainTumorModel_EffNet.h5')
    print("Model 2 saved as BrainTumorModel_EffNet.h5")

    final_epoch_eff = len(history_eff.history['loss']) - 1
    train_acc_eff  = history_eff.history['accuracy'][final_epoch_eff]
    val_acc_eff    = history_eff.history['val_accuracy'][final_epoch_eff]
    train_loss_eff = history_eff.history['loss'][final_epoch_eff]
    val_loss_eff   = history_eff.history['val_loss'][final_epoch_eff]

    print("\nMODEL 2 (EfficientNetB0) FINAL RESULTS")
    print(f"Training Accuracy:   {train_acc_eff*100:.2f}%")
    print(f"Validation Accuracy: {val_acc_eff*100:.2f}%")
    print(f"Training Loss:       {train_loss_eff:.4f}")
    print(f"Validation Loss:     {val_loss_eff:.4f}")

    plot_curves(history_eff, "EfficientNetB0", run_dir)

    # Save a small JSON file comparing both models
    metrics = {
        "CNN": {
            "Training Accuracy": round(float(train_acc_cnn), 4),
            "Validation Accuracy": round(float(val_acc_cnn), 4),
            "Training Loss": round(float(train_loss_cnn), 4),
            "Validation Loss": round(float(val_loss_cnn), 4),
        },
        "EfficientNetB0": {
            "Training Accuracy": round(float(train_acc_eff), 4),
            "Validation Accuracy": round(float(val_acc_eff), 4),
            "Training Loss": round(float(train_loss_eff), 4),
            "Validation Loss": round(float(val_loss_eff), 4),
        },
    }

    metrics_path = artifacts_root / "metrics_compare.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved comparison metrics to {metrics_path}")

    print("\nTraining plots saved in", run_dir)
