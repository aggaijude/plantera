import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
from PIL import Image

# Import predict script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "inference"))
import predict

app = FastAPI(title="Plant Health Scoring API")

# Mount the static web UI
WEB_DIR = os.path.join(BASE_DIR, "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    with open(index_path, "r") as f:
        return f.read()

@app.post("/predict")
async def predict_plant_health(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No selected file")
    
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Call the inference logic
        result = predict.predict(pil_image)
        
        # ── Null-safety sanitizer ─────────────────────────────────
        # Guarantee health_index is either a valid number or null
        hi = result.get("health_index")
        if hi is None or not isinstance(hi, (int, float)):
            result["health_index"] = None
        else:
            result["health_index"] = int(min(100, max(0, hi)))
        
        # Guarantee heatmap_url is either a string or null
        if not isinstance(result.get("heatmap_url"), str):
            result["heatmap_url"] = None
            
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return a graceful fallback instead of a 500 error 
        # so the frontend always gets a parseable JSON response
        return JSONResponse(content={
            "prediction": "Analysis Error",
            "confidence": 0.0,
            "health_index": None,
            "status": "An error occurred during analysis",
            "heatmap_url": None
        }, status_code=200)

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the server from the root of plant-ai
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
