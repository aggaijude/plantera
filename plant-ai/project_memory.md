# Plant Health Scoring AI - Project Memory

## Project Goal
Build an end-to-end Machine Learning web application that takes an image of a plant leaf and predicts a "Plant Health Score" from 0 to 100. The system uses a Deep Learning model (EfficientNet-B0 default) trained on the PlantVillage dataset using PyTorch. The backend API is powered by FastAPI, and the frontend is a simple HTML/JS/CSS web interface.

## Dataset Information
- **Source**: Kaggle PlantVillage dataset.
- **Location**: `./plant-ai/dataset/` 
- **Structure**: Folders representing plant+disease classes.
- **Data Augmentations**: Random rotation, horizontal flip, brightness variation.
- **Splits**: 70% Train, 15% Validation, 15% Test.

## Model Architecture Used
- **Base Model**: PyTorch `efficientnet_b0` (pretrained on ImageNet) or `resnet50`.
- **Modifications**: Replaced fully connected layer to output the number of our specific classes.
- **Inputs**: 224x224 Normalized Images.

## Training Parameters
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 25
- **Mixed Precision (AMP)**: Enabled for GPU Acceleration
- **DataLoader Workers**: 8 (with pin_memory=True)

## File History (Created so far)
- `plant-ai/requirements.txt`: Python package dependencies.
- `plant-ai/project_memory.md`: Persistent memory file tracking progress.
- `plant-ai/training/train_model.py`: Script to train the PyTorch model.
- `plant-ai/inference/predict.py`: Script logic to predict health score from image.
- `plant-ai/api/server.py`: FastAPI server serving endpoints.
- `plant-ai/web/index.html`, `style.css`, `script.js`: Web frontend.

## Changes Made During Development
- Initialized directory structure for project (`models/`, `training/`, `inference/`, `api/`, `web/`, `logs/`).
- Moved dataset from extracted `archive/PlantVillage` to local project space at `plant-ai/dataset/`.
- Written prediction, backend API, and web frontend UI to handle live inputs and display health score calculations explicitly mapping healthy leaves to high scores and diseased to specific lower bands.
- Fixed GPU detection so PyTorch correctly allocates the CUDA device and uses `cudnn.benchmark`.
- Added PyTorch Automatic Mixed Precision (AMP) to speed up GPU training locally.
- Optimized DataLoader instances to prevent CPU bottlenecks.
- Updated health score algorithm to naturally map healthy probability.
- Completely overhauled the Web UI into the dark/neon green 'Plantera' aesthetic featuring a robust Welcome page, glassmorphism cards, Drag & Drop inputs, and animated metric readouts using Tailwind CSS (CDN) and Vanilla JS, leaving Python endpoints fully intact.

## Model Performance Metrics
*(No model trained yet)*

## Next Tasks to Complete
- Deploy the complete Plantera ecosystem (FastAPI Backend + Static Frontend) to a production environment.
- (Optional) Port the Vanilla frontend into a robust Next.js/React project to leverage complex `21st.dev` shader components and Three.js backgrounds in the future.
