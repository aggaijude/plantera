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
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the server from the root of plant-ai
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
