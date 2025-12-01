import os
import uuid
import sqlite3
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model

# Flask + model setup
app = Flask(__name__, template_folder="templates", static_folder="static")

# Path to the trained CNN model (64x64) exported from mainTrain.py
MODEL_PATH = "BrainTumorModel.h5"

# Folder where uploaded MRI images are stored
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model once when the app starts
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded. Go to http://127.0.0.1:5000/")

# SQLite database file
DB_PATH = "neuroscan.db"


# Database helpers
def DB():
    """Open a new database connection."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Create tables if they don't exist and make sure
    the schema works for both new and old databases.
    """
    with DB() as con:
        cur = con.cursor()

        # Patients table keeps basic patient information
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Scans table stores each MRI upload and prediction
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # Helper to see if a column exists in a table
        def has_column(table, col):
            cur.execute(f"PRAGMA table_info({table})")
            return any(r[1] == col for r in cur.fetchall())

        # Older DBs may not have patient_id on scans
        if not has_column("scans", "patient_id"):
            cur.execute("ALTER TABLE scans ADD COLUMN patient_id TEXT")
            cur.execute("UPDATE scans SET patient_id = COALESCE(patient_id, 'UNKNOWN')")

        # Make sure there is always an 'UNKNOWN' patient as a fallback
        cur.execute("SELECT 1 FROM patients WHERE patient_id='UNKNOWN'")
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO patients (patient_id, name, age, sex, created_at) VALUES (?,?,?,?,?)",
                ("UNKNOWN", "Unknown Patient", None, None, datetime.utcnow().isoformat())
            )

        con.commit()


def upsert_patient(patient_id: str, name: str, age, sex: str):
    """
    If the patient exists, update basic details.
    If not, create a new patient row.
    """
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
        row = cur.fetchone()
        if row:
            existing = dict(row)
            new_name = name or existing["name"]
            new_age = age if age is not None else existing["age"]
            new_sex = sex or existing["sex"]
            cur.execute(
                "UPDATE patients SET name=?, age=?, sex=? WHERE patient_id=?",
                (new_name, new_age, new_sex, patient_id)
            )
        else:
            cur.execute(
                "INSERT INTO patients (patient_id, name, age, sex, created_at) VALUES (?, ?, ?, ?, ?)",
                (patient_id, name or "Unknown", age, sex, datetime.utcnow().isoformat())
            )
        con.commit()


def get_patient(patient_id: str):
    """Return a patient as a dict, or None if not found."""
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,))
        r = cur.fetchone()
        return dict(r) if r else None


def insert_scan(patient_id: str, filename: str, label: str, confidence: float):
    """Insert a scan row and return its ID."""
    with DB() as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO scans (patient_id, filename, label, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (patient_id, filename, label, confidence, datetime.utcnow().isoformat())
        )
        con.commit()
        return cur.lastrowid


def get_scan(scan_id: int):
    """Return a scan row by ID as a dict, or None."""
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM scans WHERE id=?", (scan_id,))
        r = cur.fetchone()
        return dict(r) if r else None


def scans_for_patient(patient_id: str):
    """Return all scans for a given patient, newest first."""
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            "SELECT * FROM scans WHERE patient_id=? ORDER BY created_at DESC",
            (patient_id,)
        )
        return [dict(r) for r in cur.fetchall()]


def all_scans(q: str = ""):
    """
    Return all scans for the History page.
    If q is non-empty, filter by patient name, ID, or label.
    """
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        if q:
            qlike = f"%{q}%"
            cur.execute("""
                SELECT s.*, p.name
                FROM scans s
                LEFT JOIN patients p ON p.patient_id = s.patient_id
                WHERE p.name LIKE ? OR s.patient_id LIKE ? OR s.label LIKE ?
                ORDER BY s.created_at DESC
            """, (qlike, qlike, qlike))
        else:
            cur.execute("""
                SELECT s.*, p.name
                FROM scans s
                LEFT JOIN patients p ON p.patient_id = s.patient_id
                ORDER BY s.created_at DESC
            """)
        return [dict(r) for r in cur.fetchall()]


def get_stats(limit_recent: int = 10):
    """
    Collect summary stats for the dashboard:
    - total scans
    - tumor / no-tumor counts
    - average confidence
    - most recent scans
    """
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        cur.execute("SELECT COUNT(*) AS c FROM scans")
        total = cur.fetchone()["c"] or 0

        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE label='Tumor'")
        tumors = cur.fetchone()["c"] or 0

        cur.execute("SELECT COUNT(*) AS c FROM scans WHERE label='No Tumor'")
        non = cur.fetchone()["c"] or 0

        cur.execute("SELECT AVG(confidence) AS a FROM scans")
        avg = cur.fetchone()["a"] or 0.0

        cur.execute("""
            SELECT id, filename, label, confidence, created_at
            FROM scans
            ORDER BY id DESC LIMIT ?;
        """, (limit_recent,))
        recent = [dict(r) for r in cur.fetchall()]
        for r in recent:
            r["thumb"] = f"static/uploads/{r['filename']}"

        return dict(
            total=int(total),
            tumors=int(tumors),
            non_tumors=int(non),
            avg_confidence=float(avg),
            recent=recent
        )


# Model helpers
def preprocess_image(image_path, target_size=(64, 64)):
    """
    Read an image from disk, convert to RGB, resize to model size,
    normalize to [0,1], and add batch dimension.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def predict_class(img_path):
    """
    Predict tumor / no-tumor for a single MRI image.
    Returns (label, confidence).
    """
    img = preprocess_image(img_path)
    pred = model.predict(img, verbose=0)[0]

    # Model can be either:
    # - single sigmoid output
    # - two-class softmax
    if len(pred) == 1:
        prob_yes = float(pred[0])
        label = "Tumor" if prob_yes >= 0.5 else "No Tumor"
        conf = prob_yes if prob_yes >= 0.5 else 1 - prob_yes
    else:
        idx = int(np.argmax(pred))
        label = "Tumor" if idx == 1 else "No Tumor"
        conf = float(pred[idx])

    return label, conf


# Routes
@app.route("/", methods=["GET"])
def dashboard():
    """Main dashboard with KPIs and recent scans."""
    stats = get_stats(limit_recent=10)
    return render_template("dashboard.html", stats=stats)


@app.route("/about", methods=["GET"])
def about():
    """Project overview / About page."""
    return render_template("about.html")


@app.route("/patients", methods=["GET", "POST"])
def patients():
    """
    GET  → show patient list and create form.
    POST → create or update a patient and (optionally) upload their first MRI.
            After POST, redirect to that patient's page with recent scan highlighted.
    """
    if request.method == "POST":
        pid = (request.form.get("patient_id") or "").strip()
        name = (request.form.get("name") or "").strip()
        age = request.form.get("age", type=int)
        sex = (request.form.get("sex") or "").strip()

        # Require at least patient ID and name
        if not pid or not name:
            with DB() as con:
                con.row_factory = sqlite3.Row
                cur = con.cursor()
                cur.execute("SELECT * FROM patients ORDER BY created_at DESC")
                plist = [dict(r) for r in cur.fetchall()]
            return render_template("patients.html", patients=plist)

        # Insert or update patient
        upsert_patient(pid, name, age, sex)

        # Optional MRI upload together with patient creation
        f = request.files.get("file")
        uploaded_filename = ""
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            nm = secure_filename(os.path.splitext(f.filename)[0])
            unique = f"{nm}_{uuid.uuid4().hex[:8]}{ext}"
            fpath = os.path.join(UPLOAD_FOLDER, unique)
            f.save(fpath)

            label, confidence = predict_class(fpath)
            insert_scan(pid, unique, label, confidence)
            uploaded_filename = unique

        # Go to patient page; if a scan was uploaded, it will be highlighted
        return redirect(url_for("patient_page", patient_id=pid, uploaded=1, file=uploaded_filename))

    # GET → list all patients
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT * FROM patients ORDER BY created_at DESC")
        plist = [dict(r) for r in cur.fetchall()]
    return render_template("patients.html", patients=plist)


@app.route("/patients/<patient_id>", methods=["GET", "POST"])
def patient_page(patient_id):
    """
    Show a single patient's details and scan history.
    POST allows uploading additional MRIs for this patient.
    """
    p = get_patient(patient_id)
    if not p:
        return f"Patient {patient_id} not found", 404

    if request.method == "POST":
        f = request.files.get("file")
        uploaded_filename = ""
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            nm = secure_filename(os.path.splitext(f.filename)[0])
            unique = f"{nm}_{uuid.uuid4().hex[:8]}{ext}"
            fpath = os.path.join(UPLOAD_FOLDER, unique)
            f.save(fpath)

            label, confidence = predict_class(fpath)
            insert_scan(patient_id, unique, label, confidence)
            uploaded_filename = unique

        # After upload, reload page and highlight the new scan
        return redirect(url_for("patient_page", patient_id=patient_id, uploaded=1, file=uploaded_filename))

    scans = scans_for_patient(patient_id)
    for s in scans:
        s["image_url"] = url_for("static", filename=f"uploads/{s['filename']}")

    just_uploaded = request.args.get("uploaded") == "1"
    just_file = (request.args.get("file") or "").strip()

    # Decide which scan to show as the large "featured" item
    featured_scan = None
    if just_file:
        for s in scans:
            if s.get("filename") == just_file:
                featured_scan = s
                break
    if not featured_scan:
        featured_scan = scans[0] if scans else None

    return render_template(
        "patient_details.html",
        patient=p,
        scans=scans,
        featured_scan=featured_scan,
        just_uploaded=just_uploaded
    )


@app.route("/history")
def history():
    """Global scan history table with optional search."""
    q = request.args.get("q", "").strip()
    items = all_scans(q)
    return render_template("history.html", items=items, q=q)


@app.route("/risk/<patient_id>")
def risk(patient_id):
    """
    Simple risk summary built from the latest scan
    (label + confidence mapped to a textual risk level).
    """
    p = get_patient(patient_id)
    if not p:
        return f"Patient {patient_id} not found", 404

    scans = scans_for_patient(patient_id)
    latest = scans[0] if scans else None

    risk_level = "Unknown"
    advice = "No scans yet."
    if latest:
        if latest["label"] == "Tumor" and latest["confidence"] >= 0.85:
            risk_level, advice = "High", "Strong tumor signal. Escalate for specialist review."
        elif latest["label"] == "Tumor":
            risk_level, advice = "Moderate", "Tumor signal present. Recommend follow-up imaging and clinical correlation."
        else:
            if latest["confidence"] >= 0.85:
                risk_level, advice = "Low", "No tumor detected with high model confidence."
            else:
                risk_level, advice = "Low–Moderate", "No tumor detected, but confidence is modest."

    return render_template("risk.html", patient=p, latest=latest, scans=scans,
                           risk_level=risk_level, advice=advice)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle uploads from the dashboard modal:
    create/update patient, run prediction, then
    redirect to the tumor details page for that scan.
    """
    patient_id = (request.form.get("patient_id") or "").strip()
    name = (request.form.get("name") or "").strip()
    age = request.form.get("age", type=int)
    sex = (request.form.get("sex") or "").strip()

    if not patient_id or 'file' not in request.files:
        return "Missing patient or file", 400

    upsert_patient(patient_id, name or "Unknown", age, sex or None)

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    ext = os.path.splitext(file.filename)[1].lower()
    nm = secure_filename(os.path.splitext(file.filename)[0])
    unique = f"{nm}_{uuid.uuid4().hex[:8]}{ext}"
    fpath = os.path.join(UPLOAD_FOLDER, unique)
    file.save(fpath)

    label, confidence = predict_class(fpath)
    scan_id = insert_scan(patient_id, unique, label, confidence)
    return redirect(url_for("tumor_details", scan_id=scan_id))


@app.route("/tumor/<int:scan_id>")
def tumor_details(scan_id):
    """
    Show one scan:
    - image
    - predicted label + confidence
    - patient info
    - simple auto note (in template)
    """
    scan = get_scan(scan_id)
    if not scan:
        return "Scan not found", 404
    patient = get_patient(scan["patient_id"])
    image_url = url_for("static", filename=f"uploads/{scan['filename']}")
    return render_template("Tumordetails.html", scan=scan, patient=patient, image_url=image_url)


@app.route("/open")
def open_gallery():
    """
    Show the latest scan for each patient as a card grid.
    Used by open.html.
    """
    with DB() as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("""
            SELECT
                p.patient_id,
                p.name,
                p.age,
                p.sex,
                s.id         AS scan_id,
                s.filename   AS filename,
                s.label      AS label,
                s.confidence AS confidence,
                s.created_at AS created_at
            FROM patients p
            JOIN scans s ON s.id = (
                SELECT s2.id
                FROM scans s2
                WHERE s2.patient_id = p.patient_id
                ORDER BY s2.created_at DESC
                LIMIT 1
            )
            ORDER BY p.created_at DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]

    cards = []
    for r in rows:
        cards.append({
            "id": r["scan_id"],
            "patient_id": r["patient_id"],
            "name": r["name"],
            "age": r["age"],
            "sex": r["sex"],
            "label": r["label"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
            "image_url": url_for("static", filename=f"uploads/{r['filename']}")
        })

    return render_template("open.html", cards=cards)


@app.route("/open/<int:scan_id>")
def open_full(scan_id):
    """
    Full-screen MRI and details view from the Open gallery.
    """
    scan = get_scan(scan_id)
    if not scan:
        return "Scan not found", 404
    patient = get_patient(scan["patient_id"])
    image_url = url_for("static", filename=f"uploads/{scan['filename']}")
    return render_template("open_full.html", scan=scan, patient=patient, image_url=image_url)


@app.route("/api/stats")
def api_stats():
    """JSON stats endpoint used by dashboard / side console."""
    return jsonify(get_stats(limit_recent=10))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
