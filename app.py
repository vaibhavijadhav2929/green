"""
GreenClassify v2: Vegetable Image Classification
Modern Flask Application with Latest TensorFlow

Author: MCA Final Year Project
Compatible with: Python 3.12+, TensorFlow 2.18+, Latest packages
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import tensorflow as tf
import logging
from werkzeug.utils import secure_filename
from pathlib import Path
import warnings

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# Flask App
app = Flask(__name__)

# Configuration
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class Mapping
CLASS_MAP = {
    0: "Bean",
    1: "Bitter Gourd",
    2: "Bottle Gourd",
    3: "Brinjal",
    4: "Broccoli",
    5: "Cabbage",
    6: "Capsicum",
    7: "Carrot",
    8: "Cauliflower",
    9: "Cucumber",
    10: "Papaya",
    11: "Potato",
    12: "Pumpkin",
    13: "Radish",
    14: "Tomato"
}

# Load Model
try:
    logger.info("Loading model...")
    
    # Try loading different model files
    model_files = [
        "vegetable_classification.h5",
        "vegetable_model_best.keras",
        "vegetable_classification.keras",
        "best.keras",
        "vegetable_model.keras",
        "vegetable_model.h5"
    ]
    
    model = None
    for model_path in model_files:
        if not os.path.exists(model_path):
            continue
            
        try:
            logger.info(f"Attempting to load: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"✓ Model loaded successfully from: {model_path}")
            logger.info(f"✓ Input shape: {model.input_shape}")
            logger.info(f"✓ Output shape: {model.output_shape}")
            break
        except Exception as e:
            logger.warning(f"  Failed to load {model_path}: {str(e)[:80]}...")
            continue
    
    if model is None:
        raise FileNotFoundError("Could not load any model file. Please check model compatibility.")

except Exception as e:
    logger.error(f"✗ Model loading failed: {e}")
    model = None

# Helper Functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Routes
@app.route("/")
def home():
    """Home page"""
    return render_template("index.html")


@app.route("/prediction")
def prediction_page():
    """Prediction page"""
    return render_template("prediction.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle prediction"""
    # Redirect GET requests to prediction page
    if request.method == "GET":
        return redirect(url_for("prediction_page"))
    
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        if "file" not in request.files:
            return redirect(url_for("prediction_page"))

        file = request.files["file"]

        if file.filename == "":
            return redirect(url_for("prediction_page"))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Preprocess and predict
            img = preprocess_image(filepath)
            predictions = model.predict(img, verbose=0)
            
            # Get top prediction
            class_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][class_index]) * 100
            vegetable = CLASS_MAP[class_index]
            
            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[::-1][:3]
            top3_predictions = [
                {
                    "name": CLASS_MAP[int(idx)],
                    "confidence": float(predictions[0][idx]) * 100
                }
                for idx in top3_indices
            ]
            
            # Low confidence warning
            low_confidence = confidence < 60

            return render_template(
                "result.html",
                vegetable=vegetable,
                confidence=f"{confidence:.2f}",
                filename=filename,
                low_confidence=low_confidence,
                top3=top3_predictions
            )

        return redirect(url_for("prediction_page"))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return redirect(url_for("prediction_page"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        predictions = model.predict(img, verbose=0)
        
        class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][class_index]) * 100
        vegetable = CLASS_MAP[class_index]

        return jsonify({
            "vegetable": vegetable,
            "confidence": confidence,
            "all_predictions": {CLASS_MAP[i]: float(predictions[0][i]) * 100 for i in range(len(CLASS_MAP))}
        })

    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__
    })


# Error Handlers
@app.errorhandler(413)
def file_too_large(e):
    return "File too large (Max 16MB)", 413


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("home"))


# Main
if __name__ == "__main__":
    # Create uploads folder
    Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("   GreenClassify v2 – Vegetable Classification System")
    print("=" * 70)
    print("✓ Flask running")
    print("✓ TensorFlow:", tf.__version__)
    print("✓ NumPy:", np.__version__)
    print("✓ Model loaded:", model is not None)
    print("✓ Classes:", len(CLASS_MAP))
    print("✓ URL: http://localhost:5000")
    print("=" * 70 + "\n")

    # Production settings
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    port = int(os.environ.get("PORT", 5000))
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
