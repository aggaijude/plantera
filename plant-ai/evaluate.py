import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, random_split
import json
from sklearn.metrics import classification_report

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "PlantVillage")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
IMG_SIZE = 224
BATCH_SIZE = 32

def main():
    print("Initializing Evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        with open(CLASSES_PATH, "r") as f:
            classes = json.load(f)
    except FileNotFoundError:
        print(f"Error: {CLASSES_PATH} not found. Please train the model first.")
        return

    print("Loading trained model...")
    model = efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
        return
        
    model.to(device)
    model.eval()

    print("Loading test dataset subset...")
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=test_transforms)
    
    # Split identically to how training split to get a "test" batch (15%)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    # Using a fixed seed generator ensures we grab a consistent split if re-run
    generator = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Running predictions on {len(test_ds)} images. This may take a moment...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "="*50)
    print("FINAL CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

if __name__ == '__main__':
    main()
