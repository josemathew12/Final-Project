import os
import uuid
import sqlite3
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
MODEL_PATH = "BrainTumorModel.h5"      # Make sure this file exists from mainTrain.py
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded. Go to http://127.0.0.1:5000/")

DB_PATH = "neuroscan.db"

# ---------------- DB UTILS ----------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        con.commit()

def insert_scan(filename: str, label: str, confidence: float):
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO scans (filename, label, confidence, created_at) VALUES (?, ?, ?, ?)",
            (filename, label, confidence, datetime.utcnow().isoformat()),
        )
        con.commit()

def get_stats(limit_recent: int = 10):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        cur.execute("SELECT COUNT(*) AS c FROM scans;")
        total = cur.fetchone()["c"] or 0

        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE label='Tumor';")
        tumors = cur.fetchone()["c"] or 0

        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE label='No Tumor';")
        non_tumors = cur.fetchone()["c"] or 0

        cur.execute("SELECT AVG(confidence) AS a FROM scans;")
        avg_conf = cur.fetchone()["a"]
        avg_conf = float(avg_conf) if avg_conf is not None else 0.0

        cur.execute(
            "SELECT filename, label, confidence, created_at FROM scans ORDER BY id DESC LIMIT ?;",
            (limit_recent,),
        )
        recent = [dict(r) for r in cur.fetchall()]

        return {
            "total": int(total),
            "tumors": int(tumors),
            "non_tumors": int(non_tumors),
            "avg_confidence": avg_conf,
            "recent": recent,
        }

# ---------------- IMAGE / PREDICTION ----------------
def preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def predict_class(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img, verbose=0)[0]
    # Works for both sigmoid (1 output) and softmax (2 outputs)
    if len(prediction) == 1:
        prob_yes = float(prediction[0])
        label = "Tumor" if prob_yes >= 0.5 else "No Tumor"
        conf = prob_yes if prob_yes >= 0.5 else 1 - prob_yes
    else:
        idx = int(np.argmax(prediction))
        label = "Tumor" if idx == 1 else "No Tumor"
        conf = float(prediction[idx])
    return label, conf

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def dashboard():
    stats = get_stats(limit_recent=10)
    return render_template("dashboard.html", stats=stats)

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/api/stats", methods=["GET"])
def api_stats():
    return jsonify(get_stats(limit_recent=10))

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # unique + safe filename
    ext = os.path.splitext(file.filename)[1].lower()
    name = secure_filename(os.path.splitext(file.filename)[0])
    unique_name = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(file_path)

    label, confidence = predict_class(file_path)
    insert_scan(unique_name, label, confidence)

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2),
        "filename": unique_name,
        "image_url": url_for('static', filename=f'uploads/{unique_name}')
    })

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
