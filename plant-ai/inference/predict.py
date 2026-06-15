import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0
from PIL import Image
import json
import cv2
import numpy as np
import torch.nn.functional as F
import base64
import threading

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "plant_model.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
IMG_SIZE = 224

# Confidence threshold — predictions below this are rejected as unsupported
CONFIDENCE_THRESHOLD = 0.40

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
        
    def __call__(self, class_idx):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        # Clone activations to avoid in-place modification on graph variables
        activations = self.activations.clone()
        # Weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # Apply ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap.cpu().detach().numpy()

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

# Global model cache and thread lock to prevent race conditions in FastAPI
try:
    GLOBAL_MODEL, GLOBAL_CLASSES = load_model()
except Exception as e:
    GLOBAL_MODEL, GLOBAL_CLASSES = None, None
    print(f"Model load error: {e}")

predict_lock = threading.Lock()

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

def calculate_health_index(predicted_class, confidence):
    """
    Computes plant health index dynamically based on severity and confidence.
    Returns an integer clamped to [0, 100], or None if inputs are invalid.
    """
    if not predicted_class or confidence is None:
        return None

    class_name_lower = predicted_class.lower()
    
    if "healthy" in class_name_lower:
        # Score 90-100 based on confidence
        base_score = 90
        score = base_score + (10 * confidence)
    elif "early" in class_name_lower or "mild" in class_name_lower:
        # Score 60-80
        score = 60 + (20 * (1 - confidence))
    else:
        # Severe disease: Score 20-50
        score = 20 + (30 * (1 - confidence))
    
    # Safety clamp to valid range
    return int(round(min(100, max(0, score))))

def predict(image_source):
    if GLOBAL_MODEL is None:
        return {"prediction": "Model Load Error", "confidence": 0, "health_index": None, "status": "Error", "heatmap_url": None}
        
    model, classes = GLOBAL_MODEL, GLOBAL_CLASSES
    
    # Needs original image for overlay
    if isinstance(image_source, str):
        original_pil = Image.open(image_source).convert('RGB')
    else:
        original_pil = image_source.convert('RGB')
        
    image_tensor = preprocess_image(original_pil)
    image_tensor.requires_grad_(True)
    
    with predict_lock:
        # Intialize GradCAM targeting the final convolution block
        cam = GradCAM(model, model.features[-1])
        
        # We purposefully exclude torch.no_grad() so we can execute backwards pass
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = classes[predicted_idx.item()]
        conf_value = float(confidence.item())
        
        # ── Confidence Rejection Gate ──────────────────────────────────
        if conf_value < CONFIDENCE_THRESHOLD:
            cam.remove_hooks()
            # Two-tier messaging for better UX
            if conf_value < 0.20:
                status_msg = "Image is likely not a plant"
            else:
                status_msg = "Unknown plant species or unsupported image"
            return {
                "prediction": "Unsupported Plant Species",
                "confidence": conf_value,
                "health_index": None,
                "status": status_msg,
                "heatmap_url": None
            }
        
        # ── Grad-CAM Heatmap Generation ───────────────────────────────
        heatmap_url = None
        try:
            model.zero_grad()
            class_score = outputs[0, predicted_idx]
            class_score.backward()
            
            heatmap_np = cam(predicted_idx)
            cam.remove_hooks() # Remove hooks after generation
            
            heatmap_resized = cv2.resize(heatmap_np, (original_pil.width, original_pil.height))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Convert original to BGR for cv2 overlay
            original_cv2 = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(original_cv2, 0.6, heatmap_colored, 0.4, 0)
            
            # Return as base64 to avoid file IO race conditions and simplify frontend integration
            _, buffer = cv2.imencode('.png', overlay)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            heatmap_url = f"data:image/png;base64,{b64_str}"
            
        except Exception as e:
            print(f"[Grad-CAM Warning] Heatmap generation failed: {e}")
            heatmap_url = None
            try:
                cam.remove_hooks()
            except:
                pass
            
        # ── Health Index Calculation ───────────────────────────────────
        health_index = calculate_health_index(predicted_class, conf_value)
        
        # Determine basic status
        status = "Healthy" if "healthy" in predicted_class.lower() else "Disease Detected"
        
        result = {
            "prediction": predicted_class.replace("_", " ").title(),
            "confidence": conf_value,
            "health_index": health_index,
            "status": status,
            "heatmap_url": heatmap_url
        }
        
        return result

if __name__ == "__main__":
    # Test stub
    # print(predict("test_image.jpg"))
    pass
