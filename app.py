"""
Food Image Classification — Flask backend using TensorFlow Lite

Project structure:
food-classifier/
├── app.py
├── seed_redis.py
├── models_tflite/
│   ├── custom_cnn.tflite
│   ├── vgg16.tflite
│   └── resnet50.tflite
├── data/
│   ├── nutritional_facts.json
│   ├── class_indices.json                 # recommended
│   ├── metrics_custom_cnn.json
│   ├── metrics_vgg16.json
│   └── metrics_resnet50.json
├── templates/index.html
└── static/...

Install (TensorFlow CPU provides tf.lite.Interpreter):
    pip install flask tensorflow-cpu numpy pillow redis gunicorn

Set REDIS_URL to your Redis connection string before starting the app.
Run:
    python seed_redis.py
    python app.py

Important:
- The preprocessing below assumes the models were trained with RGB images
  resized to 256x256 and pixel values divided by 255.0.
- Change MODEL_INPUT_SIZE or preprocessing if your training pipeline differed.
- class_indices.json must map class names to the exact output indices used
  during training, e.g. {"pizza": 0, "samosa": 1, ...}.
"""

import json
import os
import random
import threading

import numpy as np
import redis
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError


app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")  # Used by seed_redis.py

MODEL_KEYS = ("custom_cnn", "vgg16", "resnet50")

MODEL_LABELS = {
    "custom_cnn": "Custom CNN",
    "vgg16": "VGG16",
    "resnet50": "ResNet50",
}

MODEL_WEIGHTS = {
    "custom_cnn": os.path.join(MODELS_DIR, "custom_cnn.tflite"),
    "vgg16": os.path.join(MODELS_DIR, "vgg16.tflite"),
    "resnet50": os.path.join(MODELS_DIR, "resnet50.tflite"),
}

# Set these to the dimensions used when training each model.
MODEL_INPUT_SIZE = {
    "custom_cnn": (256, 256),
    "vgg16": (256, 256),
    "resnet50": (256, 256),
}

# ---------------------------------------------------------------------------
# Redis configuration
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "food_classifier")

NUTRITION_KEY = f"{REDIS_KEY_PREFIX}:nutrition_facts"
CLASS_INDICES_KEY = f"{REDIS_KEY_PREFIX}:class_indices"


def metrics_key(model_key):
    return f"{REDIS_KEY_PREFIX}:metrics:{model_key}"


redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    health_check_interval=30,
)

# ---------------------------------------------------------------------------
# TFLite model cache
# ---------------------------------------------------------------------------

# Models are loaded only when first selected. Each interpreter has its own
# lock because a TFLite Interpreter should not be invoked concurrently.
_loaded_models = {}
_model_locks = {}
_model_cache_lock = threading.Lock()


class DataUnavailableError(RuntimeError):
    """Raised when required JSON data cannot be fetched from Redis."""


def model_is_available(model_key):
    return os.path.isfile(MODEL_WEIGHTS[model_key])


def get_model(model_key):
    """Lazily load and cache one TensorFlow Lite interpreter."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model: {model_key}")

    if model_key not in _loaded_models:
        with _model_cache_lock:
            # Check again after acquiring the lock.
            if model_key not in _loaded_models:
                model_path = MODEL_WEIGHTS[model_key]

                if not os.path.isfile(model_path):
                    raise FileNotFoundError(model_path)

                interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    num_threads=2,
                )
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                if len(input_details) != 1:
                    raise RuntimeError(
                        f"{model_key} has {len(input_details)} inputs; "
                        "this app currently expects a single-input model."
                    )

                if len(output_details) != 1:
                    raise RuntimeError(
                        f"{model_key} has {len(output_details)} outputs; "
                        "this app currently expects a single-output model."
                    )

                _loaded_models[model_key] = {
                    "interpreter": interpreter,
                    "input_details": input_details,
                    "output_details": output_details,
                }
                _model_locks[model_key] = threading.Lock()

    return _loaded_models[model_key]


# ---------------------------------------------------------------------------
# Redis data helpers
# ---------------------------------------------------------------------------

def _redis_get_json(key):
    try:
        raw = redis_client.get(key)
    except redis.RedisError as exc:
        raise DataUnavailableError(
            f"Could not reach Redis for key '{key}': {exc}"
        ) from exc

    if raw is None:
        raise DataUnavailableError(
            f"Redis key '{key}' is missing. Run seed_redis.py first."
        )

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataUnavailableError(
            f"Redis key '{key}' does not contain valid JSON: {exc}"
        ) from exc


def load_nutrition():
    """Fetch nutrition facts from Redis."""
    return _redis_get_json(NUTRITION_KEY)


def load_metrics(model_key):
    """Fetch metrics for the selected model from Redis."""
    return _redis_get_json(metrics_key(model_key))


def get_class_names():
    """
    Return class names in model-output order.

    Preferred: Redis class_indices mapping, such as {"class_a": 0, "class_b": 1}.
    Fallback: alphabetical nutrition-facts keys. The fallback is only correct
    if that alphabetical order matches the model's training class order.
    """
    try:
        index_map = _redis_get_json(CLASS_INDICES_KEY)
    except DataUnavailableError:
        return sorted(load_nutrition().keys())

    if not isinstance(index_map, dict) or not index_map:
        raise DataUnavailableError(
            f"Redis key '{CLASS_INDICES_KEY}' must contain a non-empty JSON object."
        )

    try:
        indexed_names = [(int(index), name) for name, index in index_map.items()]
        indexed_names.sort(key=lambda item: item[0])

        expected_indices = list(range(len(indexed_names)))
        actual_indices = [index for index, _ in indexed_names]

        if actual_indices != expected_indices:
            raise ValueError("Class indices must be consecutive integers starting at 0.")

        return [name for _, name in indexed_names]

    except (TypeError, ValueError) as exc:
        raise DataUnavailableError(
            f"Invalid class_indices data in Redis: {exc}"
        ) from exc


def _normalize_key(key):
    """Normalize metric keys such as 'F1 Score' and 'f1_score'."""
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def find_field(data, target_key, default=None):
    """Case- and format-tolerant dictionary lookup."""
    if not isinstance(data, dict):
        return default

    target = _normalize_key(target_key)
    for key, value in data.items():
        if _normalize_key(key) == target:
            return value

    return default


# ---------------------------------------------------------------------------
# Image preprocessing and TFLite inference
# ---------------------------------------------------------------------------

def preprocess_image(file_stream, model_key, input_details):
    """
    Prepare an RGB image for the selected model.

    This implementation matches a common Keras training pipeline:
        image.resize((256, 256))
        image.astype("float32") / 255.0

    Quantized TFLite models are converted to their expected integer input
    using the tensor's quantization scale and zero point.
    """
    width, height = MODEL_INPUT_SIZE[model_key]

    with Image.open(file_stream) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)

    # IMPORTANT: Keep this normalization consistent with training.
    array = array / 255.0
    array = np.expand_dims(array, axis=0)

    input_info = input_details[0]
    input_dtype = input_info["dtype"]

    if np.issubdtype(input_dtype, np.integer):
        scale, zero_point = input_info["quantization"]

        if scale <= 0:
            raise RuntimeError(
                "The TFLite input is integer-quantized but has no valid "
                "quantization scale. Check the model conversion."
            )

        array = np.round(array / scale + zero_point)
        limits = np.iinfo(input_dtype)
        array = np.clip(array, limits.min, limits.max).astype(input_dtype)
    else:
        array = array.astype(input_dtype)

    return array


def _dequantize_output(predictions, output_info):
    """Convert quantized integer output tensors back to float values."""
    output_dtype = output_info["dtype"]

    if np.issubdtype(output_dtype, np.integer):
        scale, zero_point = output_info["quantization"]

        if scale > 0:
            predictions = (
                predictions.astype(np.float32) - zero_point
            ) * scale
        else:
            predictions = predictions.astype(np.float32)
    else:
        predictions = predictions.astype(np.float32)

    return predictions


def run_real_prediction(model_key, file_stream):
    """Run inference using the selected TensorFlow Lite model."""
    model_data = get_model(model_key)

    interpreter = model_data["interpreter"]
    input_details = model_data["input_details"]
    output_details = model_data["output_details"]

    image_array = preprocess_image(
        file_stream,
        model_key,
        input_details,
    )

    # Protect this interpreter from simultaneous requests.
    with _model_locks[model_key]:
        interpreter.set_tensor(
            input_details[0]["index"],
            image_array,
        )
        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details[0]["index"]
        )

    predictions = _dequantize_output(predictions, output_details[0])

    if predictions.ndim == 2:
        predictions = predictions[0]
    elif predictions.ndim != 1:
        raise RuntimeError(
            f"Unexpected prediction output shape: {predictions.shape}"
        )

    class_names = get_class_names()

    if len(predictions) != len(class_names):
        raise RuntimeError(
            f"Model output has {len(predictions)} classes, but Redis contains "
            f"{len(class_names)} class names. Check class_indices.json."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_class, confidence


def run_demo_prediction(model_key, filename):
    """
    Deterministic placeholder prediction for UI testing only.

    This is NOT a real model prediction and must not be used to evaluate
    model accuracy.
    """
    class_names = get_class_names()

    if not class_names:
        raise DataUnavailableError("No class names are available.")

    seed = f"{model_key}:{filename}"
    rng = random.Random(seed)
    index = rng.randrange(len(class_names))
    confidence = rng.uniform(0.62, 0.97)

    return class_names[index], confidence


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple health endpoint for deployment checks."""
    return jsonify({"status": "ok"}), 200


@app.route("/api/classes")
def api_classes():
    try:
        return jsonify({"classes": get_class_names()})
    except DataUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        app.logger.exception("Could not load class list")
        return jsonify({"error": f"Could not load class list: {exc}"}), 500


@app.route("/api/models")
def api_models():
    return jsonify({
        key: {
            "label": MODEL_LABELS[key],
            "available": model_is_available(key),
        }
        for key in MODEL_KEYS
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "Please upload an image."}), 400

    model_key = request.form.get("model")
    actual_class = request.form.get("actual_class")

    if model_key not in MODEL_KEYS:
        return jsonify({"error": "Please choose a valid model."}), 400

    if not actual_class:
        return jsonify({"error": "Please choose the actual class name."}), 400

    image_file = request.files["image"]

    try:
        # Fetch data from Redis on every prediction request.
        nutrition = load_nutrition()
        metrics = load_metrics(model_key)

        if actual_class not in nutrition:
            return jsonify({"error": f"Unknown class: {actual_class}"}), 400

        # Check that the uploaded file is a valid image.
        try:
            image_file.stream.seek(0)
            with Image.open(image_file.stream) as test_image:
                test_image.verify()
            image_file.stream.seek(0)
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({"error": "The uploaded file is not a valid image."}), 400

        demo_mode = not model_is_available(model_key)

        if demo_mode:
            predicted_class, confidence = run_demo_prediction(
                model_key,
                image_file.filename,
            )
        else:
            predicted_class, confidence = run_real_prediction(
                model_key,
                image_file.stream,
            )

        predicted_class_metrics = metrics.get(predicted_class, {})
        actual_class_metrics = metrics.get(actual_class, {})

        response = {
            "demo_mode": demo_mode,
            "model": MODEL_LABELS[model_key],
            "predicted_class": predicted_class,
            "actual_class": actual_class,
            "confidence": round(confidence * 100, 1),
            "is_correct": predicted_class == actual_class,
            "predicted_nutrition": nutrition.get(predicted_class, {}),
            "actual_nutrition": nutrition.get(actual_class, {}),
            "predicted_metrics": predicted_class_metrics,
            "actual_metrics": actual_class_metrics,
            "predicted_confusion_matrix": find_field(
                predicted_class_metrics,
                "confusion_matrix",
                {},
            ),
            "actual_confusion_matrix": find_field(
                actual_class_metrics,
                "confusion_matrix",
                {},
            ),
        }

        return jsonify(response), 200

    except DataUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except FileNotFoundError as exc:
        return jsonify({"error": f"Required model file missing: {exc}"}), 500
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    # Render/Gunicorn uses its own server command; this is for local testing.
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5002)),
    )
