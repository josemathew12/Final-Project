import os
import sys
import json
import csv
import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve
)

from tensorflow.keras.models import load_model



CLASS_NAMES = ["No Tumor", "Yes Tumor"]  # 0=no, 1=yes
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def safe_read_resize(image_path: str, input_size: int):
    """Read and resize an image safely."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is not None:
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
        except (UnidentifiedImageError, Exception):
            return None
    pil = pil.resize((input_size, input_size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def gather_images(test_root: Path):
    """Look for subfolders yes/ and no/, or infer label from filename."""
    paths = []
    have_dirs = False

    for sub in test_root.iterdir():
        if sub.is_dir():
            name = sub.name.lower()
            if name in {"no", "yes"}:
                have_dirs = True
                lbl = 0 if name == "no" else 1
                for p in sub.rglob("*"):
                    if p.is_file() and p.suffix.lower() in VALID_EXT:
                        paths.append((str(p), lbl))

    if not have_dirs:
        for p in test_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXT:
                fname = p.name.lower()
                if "yes" in fname and "no" not in fname:
                    paths.append((str(p), 1))
                elif "no" in fname and "yes" not in fname:
                    paths.append((str(p), 0))

    return paths


def ensure_outdir(base="test_results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_confusion_matrix(cm, classes, path, normalize=False):
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = cm.astype("float") / np.maximum(cm_sum, 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else "")
    )
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, y_score, roc_path, pr_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(roc_path, dpi=160, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(pr_path, dpi=160, bbox_inches="tight")
    plt.close()
    return float(roc_auc), float(pr_auc)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on a test folder.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--input_size", type=int, default=64)
    args = parser.parse_args()

    model_path = Path(args.model)
    test_root = Path(args.test_dir)
    if not model_path.exists():
        print(f"[ERR] Model not found: {model_path}")
        sys.exit(1)
    if not test_root.exists():
        print(f"[ERR] Test directory not found: {test_root}")
        sys.exit(1)

    print(f"➡️  Loading model: {model_path}")
    model = load_model(str(model_path), compile=False)

    print(f"➡️  Scanning test dir: {test_root}")
    items = gather_images(test_root)
    if not items:
        print("❌ No test images found. Add yes/ and no/ folders.")
        sys.exit(1)
    print(f"Found {len(items)} images.")

    X, y_true, fpaths = [], [], []
    for fp, lbl in items:
        arr = safe_read_resize(fp, args.input_size)
        if arr is not None:
            X.append(arr)
            y_true.append(lbl)
            fpaths.append(fp)
    if not X:
        print("❌ No valid images read.")
        sys.exit(1)

    X = np.array(X, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    print("➡️  Predicting...")
    y_prob = model.predict(X, verbose=0)
    if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
        y_score_yes = y_prob.reshape(-1)
        y_pred = (y_score_yes >= 0.5).astype(int)
    else:
        y_score_yes = y_prob[:, 1]
        y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    out_dir = ensure_outdir("test_results")
    cm_path = out_dir / "confusion_matrix_raw.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path, normalize=False)

    roc_auc = None
    pr_auc = None
    if len(np.unique(y_true)) == 2:
        roc_path = out_dir / "roc_curve.png"
        pr_path = out_dir / "pr_curve.png"
        roc_auc, pr_auc = plot_roc_pr(y_true, y_score_yes, roc_path, pr_path)
    else:
        print("⚠️  Skipping ROC/PR (only one class present).")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "accuracy": float(acc),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
            f,
            indent=2,
        )

    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {out_dir}")
    print(f"Accuracy: {acc*100:.2f}% | Precision: {p*100:.2f}% | Recall: {r*100:.2f}% | F1: {f1*100:.2f}%")

    
    try:
        webbrowser.open(f"file://{cm_path.resolve()}")
        if roc_auc is not None:
            webbrowser.open(f"file://{(out_dir / 'roc_curve.png').resolve()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

