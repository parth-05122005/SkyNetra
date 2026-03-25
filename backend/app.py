import sys
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add root to path so we can import teammates' modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.detect import detect_objects
from retrieval.search import retrieve_similar

app = FastAPI(title="SkyNetra API")

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output images so UI can display them
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

os.makedirs("temp", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "SkyNetra API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save uploaded image with unique name to avoid conflicts
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    image_path  = f"temp/{unique_name}"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Call both modules
    detection = detect_objects(image_path)
    retrieval = retrieve_similar(image_path, k=5)

    # Combine and return
    return {
        "status":    "success",
        "detection": detection,
        "retrieval": retrieval
    }