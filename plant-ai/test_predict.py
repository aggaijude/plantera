"""Quick test to debug predict function and find the actual error."""
import sys
import traceback
import json

try:
    from inference.predict import predict
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    result = predict(img)
    
    with open("test_output.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Result written to test_output.json")
    
except Exception as e:
    with open("test_output.json", "w") as f:
        json.dump({"error": str(e), "type": type(e).__name__}, f)
    traceback.print_exc()
