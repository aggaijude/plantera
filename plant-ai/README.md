# 🌿 Plantera – AI Plant Health Analyzer

Plantera is a full-stack, AI-powered web application designed to instantly diagnose plant health and detect diseases from leaf images. Using a deeply optimized **EfficientNet-B0** convolutional neural network alongside a high-performance **FastAPI** backend and a heavily animated, premium frontend, Plantera provides surgically precise health analyses in milliseconds.

## ✨ Features

- **Deep Learning AI**: Trained on the massive PlantVillage dataset to detect across 15 different plant disease classes.
- **Lightning Fast Inference**: Runs on an optimized PyTorch pipeline utilizing Automatic Mixed Precision and cuDNN auto-tuning.
- **Beautiful UI/UX**: A dark-theme, neon-green glassmorphic interface utilizing Tailwind CSS, Lucide icons, and zero-dependency SVG background animations.
- **Device Integration**: Supports direct drag-and-drop file uploads or live webcam/camera captures directly from the dashboard.

## 🛠️ Tech Stack

- **Backend**: Python, PyTorch, FastAPI, Uvicorn
- **Frontend**: Vanilla HTML5, CSS3, JavaScript, Tailwind CSS (via CDN)
- **Model Architecture**: Transfer-learned EfficientNet-B0

---

## 🚀 Getting Started

### Prerequisites
Make sure you have **Python 3.8+** installed along with `pip`.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/plantera.git
cd plantera
```

### 2. Install Dependencies
It is highly recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows use: venv\Scripts\activate
# On Mac/Linux use: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application
Start the FastAPI server (which automatically mounts the API and serves the web frontend):
```bash
python api/server.py
```
*Note: The server typically runs on `http://localhost:8000`.*

### 4. Use the App
Open your favorite browser and navigate to [http://localhost:8000](http://localhost:8000). Click **"Get Started"** to explore the dashboard and upload a leaf image to see the model in action!

---

## 📁 Project Structure

```text
plant-ai/
│
├── api/                  # FastAPI server and endpoint routing
├── inference/            # Prediction logic and health score calculations
├── models/               # Saved model weights (plant_model.pth)
├── training/             # PyTorch training algorithms and data loaders
├── web/                  # Frontend assets (index.html, style.css, script.js)
├── requirements.txt      # Python dependencies
└── README.md             # This document
```

## 🤝 Contributing
Contributions, issues, and feature requests are always welcome! Feel free to check the issues page.

## 📜 License
Distributed under the MIT License.
