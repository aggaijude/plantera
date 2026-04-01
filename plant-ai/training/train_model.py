import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.cuda.amp import autocast, GradScaler
import json

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "PlantVillage")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 25
IMG_SIZE = 224

def set_device():
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    return device

def prepare_data():
    # Define augmentations and transformations
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2), # Brightness variation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    classes = full_dataset.classes
    
    # Save classes to JSON for the predict.py script to use later
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f)
        
    print(f"Found {len(classes)} classes.")

    # 70% Train, 15% Val, 15% Test Split
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    # We apply the specific transformations
    train_ds.dataset.transform = data_transforms
    val_ds.dataset.transform = val_test_transforms
    test_ds.dataset.transform = val_test_transforms
    
    print(f"Dataset Split: Train={train_size}, Val={val_size}, Test={test_size}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader, len(classes)

def create_model(num_classes, device):
    # Load pretrained EfficientNet-B0
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace fully connected layer (classifier)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device)

def train_model():
    device = set_device()
    train_loader, val_loader, test_loader, num_classes = prepare_data()
    
    model = create_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    best_val_acc = 0.0

    print("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1}/{EPOCHS}\n"
              f"Train Loss: {train_loss:.2f}\n"
              f"Validation Loss: {val_loss:.2f}\n"
              f"Accuracy: {val_acc*100:.0f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"--> Saved best model with Val Acc: {val_acc:.4f}")

    print("Training Completed.")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    train_model()
