"""
╔══════════════════════════════════════════════════════════╗
║       PLANT HEALTH API — Flask Backend                   ║
╚══════════════════════════════════════════════════════════╝
Run with:  python server.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import os, json, random, io, time

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")
CORS(app)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

def _load_classes():
    for candidate in ["saved_models/class_names.json", "class_names.json"]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return [
        "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
        "Apple___healthy","Blueberry___healthy","Cherry___Powdery_mildew",
        "Cherry___healthy","Corn___Cercospora_leaf_spot","Corn___Common_rust",
        "Corn___Northern_Leaf_Blight","Corn___healthy","Grape___Black_rot",
        "Grape___Esca_(Black_Measles)","Grape___Leaf_blight","Grape___healthy",
        "Orange___Haunglongbing","Peach___Bacterial_spot","Peach___healthy",
        "Pepper___Bacterial_spot","Pepper___healthy","Potato___Early_blight",
        "Potato___Late_blight","Potato___healthy","Raspberry___healthy",
        "Soybean___healthy","Squash___Powdery_mildew","Strawberry___Leaf_scorch",
        "Strawberry___healthy","Tomato___Bacterial_spot","Tomato___Early_blight",
        "Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites","Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]

CLASSES = _load_classes()

TREATMENT_MAP = {
    "healthy":          ("Healthy","No treatment needed. Keep up the good care!","healthy"),
    "Apple_scab":       ("Mild-Moderate","Apply fungicide (captan/mancozeb). Prune infected leaves.","mild"),
    "Black_rot":        ("Moderate","Remove infected parts. Apply copper-based fungicide.","moderate"),
    "Late_blight":      ("Severe","Apply chlorothalonil immediately. Remove infected plants.","severe"),
    "Early_blight":     ("Mild-Moderate","Improve air circulation. Apply azoxystrobin fungicide.","mild"),
    "Leaf_Mold":        ("Mild","Reduce humidity. Apply copper hydroxide fungicide.","mild"),
    "Spider_mites":     ("Moderate","Apply miticide/neem oil. Increase plant humidity.","moderate"),
    "Bacterial_spot":   ("Moderate","Apply copper bactericide. Avoid overhead watering.","moderate"),
    "Powdery_mildew":   ("Mild-Moderate","Apply sulfur-based fungicide. Improve air circulation.","mild"),
    "Mosaic_virus":     ("Severe","No cure. Remove infected plants. Control aphid vectors.","severe"),
    "Yellow_Leaf_Curl": ("Severe","No cure. Remove plants. Control whitefly vectors.","severe"),
    "Haunglongbing":    ("Critical","No cure. Remove tree immediately. Control psyllid insects.","critical"),
    "Cedar_apple_rust": ("Moderate","Apply myclobutanil fungicide. Remove cedar galls nearby.","moderate"),
    "Cercospora":       ("Moderate","Apply fungicides containing azoxystrobin or pyraclostrobin.","moderate"),
    "Common_rust":      ("Moderate","Apply fungicide at first sign. Plant resistant varieties.","moderate"),
    "Northern_Leaf":    ("Moderate","Apply propiconazole fungicide. Rotate crops annually.","moderate"),
    "Esca":             ("Severe","Prune infected wood. Apply wound sealants. No full cure.","severe"),
    "Leaf_blight":      ("Moderate","Apply copper-based fungicide. Remove affected leaves.","moderate"),
    "Leaf_scorch":      ("Moderate","Improve drainage. Apply copper fungicide if bacterial.","moderate"),
    "Septoria":         ("Moderate","Remove infected leaves. Apply chlorothalonil fungicide.","moderate"),
    "Target_Spot":      ("Moderate","Apply azoxystrobin fungicide. Improve plant spacing.","moderate"),
    "default":          ("Unknown","Consult a plant pathologist for accurate diagnosis.","mild"),
}

IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

_model = None
_model_loaded = False
_model_img_size = IMG_SIZE

def _init_model():
    global _model, _model_loaded, _model_img_size
    model_paths = [
        "saved_models/plant_health_lite.keras",
        "saved_models/best_model.keras",
        "plant_health_lite.keras",
    ]
    try:
        import tensorflow as tf
        for path in model_paths:
            if os.path.exists(path):
                _model = tf.keras.models.load_model(path)
                _model_loaded = True
                try:
                    in_shape = _model.input_shape
                    if isinstance(in_shape, list):
                        in_shape = in_shape[0]
                    if in_shape and len(in_shape) >= 3 and in_shape[1] is not None:
                        _model_img_size = (in_shape[1], in_shape[2])
                except Exception:
                    pass
                print(f"✅ Model loaded from {path} (img_size={_model_img_size})")
                return
        print("⚠️  No model found — running in demo mode")
    except ImportError:
        print("⚠️  TensorFlow not available — running in demo mode")
    except Exception as e:
        print(f"⚠️  Could not load model: {e} — running in demo mode")

_init_model()

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def preprocess_image(image, size=None):
    """Match the training pipeline exactly:
    - image_dataset_from_directory uses tf.image.resize (bilinear)
    - Training used Rescaling(1.0) → pixels stay [0, 255]
    - MobileNetV3Small applies its own internal preprocessing
    """
    sz = size or IMG_SIZE
    img = image.convert("RGB")

    # Center-crop to square (matches training data which is already square)
    w, h = img.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

    # Use TF resize with bilinear interpolation to match training pipeline
    try:
        import tensorflow as tf
        arr = np.array(img, dtype=np.float32)
        arr = tf.image.resize(arr, sz, method="bilinear").numpy()
    except Exception:
        img = img.resize(sz, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)

    return np.expand_dims(arr, axis=0)

def real_predict(image, top_k=5):
    arr = preprocess_image(image, size=_model_img_size)
    preds = _model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]
    return [
        {"class": CLASSES[i], "confidence": float(preds[i]), "rank": r + 1}
        for r, i in enumerate(top_idx)
    ]

def demo_predict(image, plant_hint="auto", top_k=5):
    time.sleep(0.5)
    pool = CLASSES
    if plant_hint != "auto":
        filtered = [c for c in CLASSES if c.lower().startswith(plant_hint.lower())]
        if filtered:
            pool = filtered
    primary = random.choice(pool)
    primary_conf = random.uniform(0.72, 0.97)
    others = [c for c in CLASSES if c != primary]
    random.shuffle(others)
    remaining = 1.0 - primary_conf
    results = [{"class": primary, "confidence": primary_conf, "rank": 1}]
    for r, cls in enumerate(others[:top_k - 1], 2):
        conf = remaining * random.uniform(0.05, 0.5)
        remaining -= conf
        results.append({"class": cls, "confidence": max(conf, 0.001), "rank": r})
    results.sort(key=lambda x: -x["confidence"])
    for r, p in enumerate(results, 1):
        p["rank"] = r
    return results

def build_report(predictions):
    primary = predictions[0]
    cls = primary["class"]
    parts = cls.split("___")
    plant = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in cls.lower()
    key = "healthy" if is_healthy else next(
        (k for k in TREATMENT_MAP if k.lower() in cls.lower().replace(" ", "_")),
        "default"
    )
    severity, recommendation, sev_cls = TREATMENT_MAP[key]
    return {
        "status": "HEALTHY" if is_healthy else "DISEASED",
        "plant": plant,
        "disease": disease,
        "confidence": primary["confidence"],
        "severity": severity,
        "severityClass": sev_cls,
        "recommendation": recommendation,
        "predictions": predictions,
        "isHealthy": is_healthy,
    }

# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/api/info")
def api_info():
    return jsonify({
        "modelLoaded": _model_loaded,
        "classes": CLASSES,
        "numClasses": len(CLASSES),
        "imgSize": list(_model_img_size),
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    top_k = int(request.form.get("top_k", 5))
    plant_hint = request.form.get("plant_hint", "auto")
    try:
        image = Image.open(io.BytesIO(file.read()))
        if _model_loaded:
            predictions = real_predict(image, top_k=top_k)
        else:
            predictions = demo_predict(image, plant_hint, top_k=top_k)
        report = build_report(predictions)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve React frontend
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")
