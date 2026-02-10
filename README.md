# 🥬 GreenClassify v2

**Latest Technology Stack - Production Ready**

Modern vegetable classification system using deep learning with the latest packages and best practices.

## 🚀 What's New in v2

### Updated Technology Stack
- **TensorFlow 2.18+** - Latest stable version
- **NumPy 2.0+** - Improved performance
- **Flask 3.1+** - Latest web framework
- **Python 3.12+** - Modern Python features

### New Features
- ✅ Modern .keras model format support (with .h5 fallback)
- ✅ RESTful API endpoint (`/api/predict`)
- ✅ Health check endpoint
- ✅ Top 3 predictions display
- ✅ Low confidence warnings
- ✅ Drag & drop file upload
- ✅ Real-time image preview
- ✅ Responsive modern UI
- ✅ Better error handling

## 📦 Installation

### 1. Create Virtual Environment

```bash
cd collegeproject_v2
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\\Scripts\\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🎯 Running the Application

```bash
python app.py
```

Then open: http://localhost:5000

## 🔧 API Usage

### Predict Endpoint

```bash
curl -X POST http://localhost:5000/api/predict \\
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "vegetable": "Tomato",
  "confidence": 95.67,
  "all_predictions": {
    "Bean": 0.23,
    "Tomato": 95.67,
    ...
  }
}
```

### Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tensorflow_version": "2.18.0"
}
```

## 📁 Model File

Place your trained model in the project root:
- **Preferred:** `vegetable_model.keras` (new format)
- **Legacy:** `vegetable_model.h5` (fallback)

## 🌟 Supported Vegetables

1. Bean
2. Bitter Gourd
3. Bottle Gourd
4. Brinjal
5. Broccoli
6. Cabbage
7. Capsicum
8. Carrot
9. Cauliflower
10. Cucumber
11. Papaya
12. Potato
13. Pumpkin
14. Radish
15. Tomato

## 🚢 Deployment

### Render / Railway

```bash
# Files included:
# - requirements.txt
# - Procfile (for Gunicorn)
# - runtime.txt (Python version)
```

### Docker (Optional)

```bash
docker build -t greenclassify-v2 .
docker run -p 5000:5000 greenclassify-v2
```

## 📊 Project Structure

```
collegeproject_v2/
├── app.py              # Main Flask application
├── requirements.txt    # Dependencies
├── README.md          # Documentation
├── templates/         # HTML templates
│   ├── index.html
│   ├── prediction.html
│   └── result.html
├── static/            # Static files
├── uploads/           # Uploaded images
└── venv/             # Virtual environment
```
## 🎨 Design

This section outlines the system design, architecture, and major components of GreenClassify v2.

- **Architecture:** Simple Flask web server hosting a TensorFlow model that serves predictions through a REST API. Clients (browser or curl) upload images to the API which returns top-N predictions.
- **Components:**
  - **Data pipeline:** Images organized into `train/`, `validation/`, `test/` folders under `Vegetable_Dataset/`. Preprocessing includes resizing to model input size, normalization, and optional augmentation during training.
  - **Model:** Transfer learning based on ResNet50 (fine-tuned). Preferred model file: `vegetable_model.keras` with fallback `vegetable_model.h5`.
  - **Inference:** `app.py` loads the model at startup, exposes `/api/predict` and `/health` endpoints, and returns top-3 predictions with confidence scores.
  - **Web UI:** Simple responsive frontend under `templates/` with drag-and-drop upload, real-time preview, and result pages.
  - **Uploads storage:** Uploaded images stored in `uploads/` for debugging and optional re-training.
  - **Testing & Validation:** Unit tests in `test_*.py` and sample notebooks in `backup/` for reproducibility and evaluation.

- **Deployment:** Can run on Heroku/Render/Railway or inside Docker. Use `Procfile` and `requirements.txt` for platform deployment and `docker` for containerization.
- **Scalability & Performance:** For production, serve the model with a WSGI server (Gunicorn) and place behind a reverse proxy or load balancer; consider using TensorFlow Serving or model quantization for lower latency.
- **Security & Privacy:** Validate uploads, set reasonable file-size limits, and avoid logging sensitive user data. Store models and uploads with appropriate permissions.
- **Extensibility:** Add new classes by retraining with images added to `Vegetable_Dataset/` and exporting a new `.keras` model. Training scripts are in `train_model.py` and `train_simple.py`.

## 🎓 Academic Information

- **Project:** MCA Final Year
- **Version:** 2.0 (Latest Stack)
- **Framework:** TensorFlow 2.18+
- **Architecture:** ResNet50 Transfer Learning

## 📝 License

MIT License

---

**Made with ❤️ using latest technology**
