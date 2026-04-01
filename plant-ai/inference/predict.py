import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0
from PIL import Image
import json

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
IMG_SIZE = 224

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Missing model or classes file in {MODELS_DIR}")
        
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
        
    model = efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model, classes

def preprocess_image(image_file):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        # Assuming it's already a PIL Image from the API
        image = image_file.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0).to(device)

def calculate_health_score(predicted_class, class_probabilities, classes):
    """
    Computes plant health score based on probability.
    health_score = probability_of_healthy_class * 100
    If disease classes dominate, the healthy probability naturally approaches 0,
    reducing the score accordingly.
    """
    # Find indices for all 'healthy' classes
    healthy_indices = [i for i, c in enumerate(classes) if 'healthy' in c.lower()]
    
    if not healthy_indices:
        # Fallback if no healthy class exists (shouldn't happen with PlantVillage)
        return 0
        
    # Sum the probabilities of all healthy classes
    healthy_prob_sum = sum(class_probabilities[0][i].item() for i in healthy_indices)
    
    # Calculate score
    health_score = healthy_prob_sum * 100
    
    # Cap between 0 and 100 and round
    return max(0, min(100, round(health_score)))

def predict(image_source):
    model, classes = load_model()
    image_tensor = preprocess_image(image_source)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = classes[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    
    health_score = calculate_health_score(predicted_class, probabilities, classes)
    
    result = {
        "prediction": predicted_class.replace("_", " ").title(),
        "health_score": health_score,
        "confidence": confidence_pct
    }
    
    return result

if __name__ == "__main__":
    # Test stub
    # print(predict("test_image.jpg"))
    pass
