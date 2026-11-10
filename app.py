"""
Food Image Classification — Flask backend
==========================================

Expected project layout
------------------------
food-classifier/
├── app.py
├── seed_redis.py           <- one-off script: loads data/*.json into Redis
├── models/
│   ├── custom_cnn.h5      <- your trained Keras/TensorFlow weight files
│   ├── vgg16.h5
│   └── resnet50.h5
├── data/                   <- source-of-truth JSON files, used only to SEED
│   │                          Redis (see seed_redis.py). At request time the
│   │                          app reads exclusively from Redis, not disk.
│   ├── nutritional_facts.json   <- {"Class Name": {"calories": ..., ...}, ...}
│   ├── metrics_custom_cnn.json  <- {"Class Name": {"accuracy": ..., "precision": ..., "recall": ...,
│   │                                  "f1_score": ..., "confusion_matrix": {"tp": ..., "fp": ...,
│   │                                  "fn": ..., "tn": ...}}, ...}
│   │                              (confusion_matrix is optional, one-vs-rest counts for that class;
│   │                               omit it and the UI simply shows "no confusion matrix data")
│   ├── metrics_vgg16.json
│   ├── metrics_resnet50.json
│   └── class_indices.json       <- OPTIONAL {"Class Name": index, ...} matching the order
│                                    your model was trained with (e.g. Keras
│                                    ImageDataGenerator.class_indices). If this file
│                                    is missing, class names are taken from
│                                    nutritional_facts.json sorted alphabetically —
│                                    make sure that matches your model's output order!
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/script.js

Redis
-----
Nutrition facts and per-model metrics/confusion-matrix data are stored in
Redis and fetched fresh every time a user hits "Predict" (no on-disk fallback
and no long-lived in-process cache for this data, so updates pushed into
Redis show up immediately). Configure the connection with:

    REDIS_URL=redis://localhost:6379/0      (default if unset)
    REDIS_KEY_PREFIX=food_classifier         (default if unset)

Keys read at request time (all JSON-encoded strings):
    {prefix}:nutrition_facts        -> {"Class Name": {...}, ...}
    {prefix}:class_indices          -> {"Class Name": index, ...}   (optional)
    {prefix}:metrics:<model_key>    -> {"Class Name": {...}, ...}   one per MODEL_KEYS

Run `python seed_redis.py` once to load your existing data/*.json files into
Redis under these keys (safe to re-run any time the JSON files change).

Run
---
    pip install flask tensorflow pillow numpy redis
    python seed_redis.py   # first time only, or whenever data/*.json changes
    python app.py

The app starts in DEMO MODE automatically for any model whose .h5 weight file
is missing, so the frontend can be exercised end-to-end before real weights are
dropped in. Swap in real files under models/ and demo mode turns itself off.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import gc
import json
import os
import random
import numpy as np
import redis
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")  # only used by seed_redis.py now

MODEL_KEYS = ("custom_cnn", "vgg16", "resnet50")

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "food_classifier")

NUTRITION_KEY = f"{REDIS_KEY_PREFIX}:nutrition_facts"
CLASS_INDICES_KEY = f"{REDIS_KEY_PREFIX}:class_indices"


def metrics_key(model_key):
    return f"{REDIS_KEY_PREFIX}:metrics:{model_key}"


# A single shared connection pool. decode_responses=True so GET returns str,
# not bytes, which keeps json.loads() simple below.
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

MODEL_LABELS = {
    "custom_cnn": "Custom CNN",
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
}

MODEL_WEIGHTS = {
    "custom_cnn": os.path.join(MODELS_DIR, "custom_cnn.keras"),
    "vgg16": os.path.join(MODELS_DIR, "vgg16.keras"),
    "resnet50": os.path.join(MODELS_DIR, "resnet50.keras"),
}

# Input size each architecture expects. Adjust to match how you trained them.
MODEL_INPUT_SIZE = {
    "custom_cnn": (256, 256),
    "vgg16": (256, 256),
    "resnet50": (256, 256),
}

# Only the (heavy) Keras models get an in-process cache — loading a .keras
# file from disk is expensive and models don't change between requests.
_loaded_model = None
_loaded_model_key = None


class DataUnavailableError(RuntimeError):
    """Raised when required JSON data can't be fetched from Redis."""


# ---------------------------------------------------------------------------
# Data loading helpers — everything below reads straight from Redis, fresh,
# on every call. There is deliberately no in-process cache for this data: it
# is small, Redis is fast, and this guarantees a user hitting "Predict" (or
# hitting /api/classes, /api/models) always sees whatever is currently in
# Redis, even if it was updated moments ago.
# ---------------------------------------------------------------------------
def _redis_get_json(key):
    try:
        raw = redis_client.get(key)
    except redis.RedisError as exc:
        raise DataUnavailableError(f"Could not reach Redis for key '{key}': {exc}") from exc

    if raw is None:
        raise DataUnavailableError(f"Redis key '{key}' is missing. Run seed_redis.py first.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataUnavailableError(f"Redis key '{key}' does not contain valid JSON: {exc}") from exc


def load_nutrition():
    """Fetch the full nutrition-facts dict from Redis."""
    return _redis_get_json(NUTRITION_KEY)


def load_metrics(model_key):
    """Fetch the per-class metrics dict for one model from Redis."""
    return _redis_get_json(metrics_key(model_key))


def get_class_names():
    """Ordered list of class names, index-aligned with model output.

    Prefers the {prefix}:class_indices Redis key (explicit, safest). Falls
    back to the alphabetically-sorted keys of the nutrition-facts data.
    """
    try:
        idx_map = _redis_get_json(CLASS_INDICES_KEY)
    except DataUnavailableError:
        return sorted(load_nutrition().keys())

    ordered = [None] * len(idx_map)
    for name, idx in idx_map.items():
        ordered[int(idx)] = name
    return ordered


def _normalize_key(key):
    """Lowercase and collapse spaces/hyphens to underscores, so 'F1 Score',
    'f1-score', and 'f1_score' all compare equal."""
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def find_field(d, target_key, default=None):
    """Case/format-tolerant dict lookup. Real-world metrics JSON is rarely
    consistent about 'confusion_matrix' vs 'Confusion Matrix' vs
    'confusion-matrix', so match on a normalized key instead of an exact
    one. Falls back to `default` (mutated freshly each call, not shared)
    if nothing matches."""
    if not isinstance(d, dict):
        return default
    target_norm = _normalize_key(target_key)
    for key, value in d.items():
        if _normalize_key(key) == target_norm:
            return value
    return default


def model_is_available(model_key):
    return os.path.exists(MODEL_WEIGHTS[model_key])



def get_model(model_key):
    global _loaded_model, _loaded_model_key

    # Reuse the model if it is already loaded
    if _loaded_model is not None and _loaded_model_key == model_key:
        return _loaded_model

    path = MODEL_WEIGHTS[model_key]

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Release the previously loaded model
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
        _loaded_model_key = None

        tf.keras.backend.clear_session()
        gc.collect()

    # Load only the newly selected model
    _loaded_model = load_model(path)
    _loaded_model_key = model_key

    return _loaded_model


def preprocess_image(file_stream, model_key):
    from PIL import Image

    size = MODEL_INPUT_SIZE[model_key]
    img = Image.open(file_stream).convert("RGB").resize(size)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def run_real_prediction(model_key, file_stream):

    model = get_model(model_key)
    x = preprocess_image(file_stream, model_key)
    preds = model.predict(x, verbose=0)[0]
    class_names = get_class_names()
    idx = int(np.argmax(preds))
    return class_names[idx], float(preds[idx])


def run_demo_prediction(model_key, filename):
    """Deterministic pseudo-prediction used when a model's .h5 file isn't
    present yet, so the UI/UX can be reviewed before training finishes."""
    class_names = get_class_names()
    seed = f"{model_key}:{filename}"
    rng = random.Random(seed)
    idx = rng.randrange(len(class_names))
    confidence = rng.uniform(0.62, 0.97)
    return class_names[idx], confidence


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classes")
def api_classes():
    try:
        return jsonify({"classes": get_class_names()})
    except DataUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Could not load class list: {exc}"}), 500


@app.route("/api/models")
def api_models():
    return jsonify(
        {
            key: {
                "label": MODEL_LABELS[key],
                "available": model_is_available(key),
            }
            for key in MODEL_KEYS
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "Please upload an image."}), 400

    model_key = request.form.get("model")
    actual_class = request.form.get("actual_class")

    if model_key not in MODEL_KEYS:
        return jsonify({"error": "Please choose a model."}), 400
    if not actual_class:
        return jsonify({"error": "Please choose the actual class name."}), 400

    image_file = request.files["image"]

    try:
        # Pulled fresh from Redis on every click of "Predict".
        nutrition = load_nutrition()
        metrics = load_metrics(model_key)

        if actual_class not in nutrition:
            return jsonify({"error": f"Unknown class: {actual_class}"}), 400

        demo_mode = not model_is_available(model_key)
        if demo_mode:
            predicted_class, confidence = run_demo_prediction(
                model_key, image_file.filename
            )
        else:
            predicted_class, confidence = run_real_prediction(
                model_key, image_file.stream
            )

        response = {
            "demo_mode": demo_mode,
            "model": MODEL_LABELS[model_key],
            "predicted_class": predicted_class,
            "actual_class": actual_class,
            "confidence": round(confidence * 100, 1),
            "is_correct": predicted_class == actual_class,
            "predicted_nutrition": nutrition.get(predicted_class, {}),
            "actual_nutrition": nutrition.get(actual_class, {}),
            "predicted_metrics": metrics.get(predicted_class, {}),
            "actual_metrics": metrics.get(actual_class, {}),
            "predicted_confusion_matrix": find_field(
                metrics.get(predicted_class, {}), "confusion_matrix", {}
            ),
            "actual_confusion_matrix": find_field(
                metrics.get(actual_class, {}), "confusion_matrix", {}
            ),
        }
        return jsonify(response)

    except DataUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except FileNotFoundError as exc:
        return jsonify({"error": f"Required model file missing: {exc}"}), 500
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5002)
