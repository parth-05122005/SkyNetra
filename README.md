# 🛰️ SkyNetra — Satellite Visual Search, Retrieval and Detection System

SkyNetra is an end-to-end modular AI pipeline that automatically **detects objects**, **generates labels**, **retrieves similar images**, and **clusters unknown objects** in satellite imagery using a hybrid supervised + unsupervised learning approach.

---

## 📌 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Download Model Weights](#-download-model-weights)
- [Running the Project](#-running-the-project)
- [How to Use](#-how-to-use)
- [API Reference](#-api-reference)
- [How Semi-Supervised Learning Works](#-how-semi-supervised-learning-works)
- [Module Overview](#-module-overview)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- 🔍 **Object Detection** — YOLOv8 fine-tuned on 14-class satellite dataset
- 🏷️ **Auto Label Generation** — Automatically generates YOLO format `.txt` annotation files
- 🔁 **Visual Similarity Search** — ResNet50 + FAISS returns top-5 similar satellite images
- 🧠 **Semi-Supervised Learning** — Supervised for known objects, unsupervised K-Means clustering for unknown objects
- ⚡ **FastAPI Backend** — Parallel processing of detection and retrieval
- 🖥️ **Streamlit UI** — Clean upload interface with downloadable results

---

## 📁 Project Structure

```
SkyNetra/
├── detection/
│   ├── __init__.py
│   └── detect.py              ← YOLOv8 detection + auto-labelling + semi-supervised
├── retrieval/
│   ├── __init__.py
│   ├── feature_extractor.py   ← ResNet50 feature extraction
│   ├── faiss_index.py         ← Build FAISS index from dataset
│   ├── search.py              ← Similarity search
│   ├── cluster.py             ← K-Means unsupervised clustering
│   └── embeddings/            ← Stores FAISS index + cluster model (auto-generated)
├── backend/
│   └── app.py                 ← FastAPI backend
├── ui/
│   └── app.py                 ← Streamlit frontend
├── data/
│   └── images/                ← Place your satellite images here
├── outputs/                   ← Annotated images + label files (auto-generated)
├── weights/
│   └── best.pt                ← Fine-tuned YOLOv8 model (download separately)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧰 Tech Stack

| Component               | Technology             |
| ----------------------- | ---------------------- |
| Object Detection        | YOLOv8 (Ultralytics)   |
| Feature Extraction      | ResNet50 (PyTorch)     |
| Similarity Search       | FAISS                  |
| Unsupervised Clustering | K-Means (Scikit-learn) |
| Backend API             | FastAPI                |
| Frontend UI             | Streamlit              |
| Image Processing        | OpenCV                 |
| Language                | Python 3.x             |

---

## ✅ Prerequisites

Make sure you have the following installed on your system:

- Python 3.8 or above
- Git
- pip

Check your Python version:

```bash
python --version
```

---

## ⚙️ Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/SkyNetra.git
cd SkyNetra
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv/Scripts/activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download Model Weights

The fine-tuned model `best.pt` is not included in the repository due to file size. Download it from the link below and place it inside the `weights/` folder.

**👉 [Download best.pt from Google Drive](https://drive.google.com/file/d/1txnTGK_P3BGdg87SojwjSx5fmZ3zJaBr/view?usp=drive_link)**

After downloading:

```
SkyNetra/
└── weights/
    └── best.pt    ← place it here
```

---

## 🚀 Running the Project

You need to run **3 steps** before starting the app.

---

### Step 1 — Add satellite images to the dataset folder

Place your satellite images (`.jpg` or `.png`) inside:

```
data/images/
```

The more images you add, the better the retrieval and clustering results.

---

### Step 2 — Build the FAISS index

This indexes all images in `data/images/` for similarity search:

```bash
python retrieval/faiss_index.py
```

You should see:

```
Processing images...
✅ FAISS index saved successfully.
```

---

### Step 3 — Train the unsupervised clusters

This trains K-Means on your image dataset for unknown object clustering:

```bash
python retrieval/cluster.py
```

You should see:

```
Extracting features from X images...
Training KMeans with 10 clusters...
✅ Done! 10 unsupervised clusters created.
```

---

### Step 3.1 — Make sure that you have created an empty output folder

### Step 4 — Start the FastAPI backend

Open **Terminal 1** and run:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

You should see:

```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
✅ FAISS index loaded successfully.
```

---

### Step 5 — Start the Streamlit UI

Open **Terminal 2** and run:

```bash
python -m streamlit run ui/app.py
```

You should see:

```
Local URL: http://localhost:8501
```

---

### Step 6 — Open in browser

Go to 👉 **http://localhost:8501**

---

## 🖥️ How to Use

1. Open **http://localhost:8501** in your browser
2. Click **Browse files** and upload a satellite image
3. Click **🔍 Analyze Image**
4. View results:
   - **Detected Objects** — labels, confidence scores, bounding boxes
   - **Learning Mode** — supervised (green) or unsupervised (orange)
   - **Similar Images** — top-5 visually similar satellite images
   - **Download Label File** — YOLO format `.txt` annotation file

> ⚠️ **Important:** Always upload satellite/aerial images for best results. The model is trained specifically on satellite imagery.

---

## 📡 API Reference

The FastAPI backend runs at `http://localhost:8000`

### Health Check

```
GET http://localhost:8000/
```

Response:

```json
{
  "status": "SkyNetra API is running"
}
```

### Analyze Image

```
POST http://localhost:8000/analyze
Content-Type: multipart/form-data
Body: file (image)
```

Response:

```json
{
  "status": "success",
  "detection": {
    "objects": [
      {
        "label": "City",
        "confidence": 0.91,
        "bbox": [x1, y1, x2, y2],
        "learning_type": "supervised",
        "cluster_id": null
      }
    ],
    "annotated_image": "outputs/detected.jpg",
    "label_file": "outputs/sample.txt",
    "summary": {
      "total_objects": 1,
      "supervised_count": 1,
      "unsupervised_count": 0
    }
  },
  "retrieval": {
    "results": [
      {
        "image": "data/images/img01.jpg",
        "similarity_score": 0.95
      }
    ]
  }
}
```

You can also test the API directly at:
👉 **http://localhost:8000/docs** (Swagger UI)

---

## 🧠 How Semi-Supervised Learning Works

SkyNetra uses a hybrid learning approach:

```
Upload Image
     ↓
YOLOv8 Detection
     ↓
Confidence > 60%?
   ↙           ↘
  YES            NO
   ↓              ↓
SUPERVISED    UNSUPERVISED
YOLOv8 label  K-Means cluster
Green bbox    Orange bbox
```

| Condition        | Method                 | Output                        | Box Color |
| ---------------- | ---------------------- | ----------------------------- | --------- |
| Confidence ≥ 60% | Supervised (YOLOv8)    | Known label e.g. `City`       | 🟩 Green  |
| Confidence < 60% | Unsupervised (K-Means) | `Unknown Pattern (Cluster 3)` | 🟧 Orange |

---

## 📦 Module Overview

### Module 1 — Detection + Auto Labelling

- Loads fine-tuned YOLOv8 model from `weights/best.pt`
- Runs prediction with confidence threshold
- Applies semi-supervised logic
- Draws bounding boxes on image
- Saves annotated image to `outputs/detected.jpg`
- Generates YOLO label file at `outputs/sample.txt`

```python
from detection.detect import run_detection

result = run_detection("data/images/sample.jpg")
print(result)
```

---

### Module 2 — Visual Similarity Search

- Extracts 2048-dim feature vector using ResNet50
- Searches FAISS index for nearest neighbors
- Returns top-K similar images with scores

```python
from retrieval.search import retrieve_similar

result = retrieve_similar("data/images/sample.jpg", k=5)
print(result)
```

---

### Module 3 — Backend + UI

- FastAPI `/analyze` endpoint calls both modules in parallel
- Streamlit UI for upload, display and download

---

## 🔧 Troubleshooting

| Error                          | Fix                                               |
| ------------------------------ | ------------------------------------------------- |
| `Module not found`             | Run `pip install -r requirements.txt`             |
| `Could not connect to backend` | Make sure FastAPI is running in Terminal 1        |
| `weights/best.pt not found`    | Download `best.pt` and place in `weights/` folder |
| `Index not built`              | Run `python retrieval/faiss_index.py` first       |
| `No images found`              | Add images to `data/images/` folder               |
| `FAISS index not found`        | Run `python retrieval/faiss_index.py`             |
| `Cluster model not found`      | Run `python retrieval/cluster.py`                 |
| Empty detection results        | Use actual satellite/aerial images                |

---

## 📋 Requirements

```
ultralytics
opencv-python
torch
torchvision
faiss-cpu
numpy
fastapi
uvicorn
streamlit
requests
python-multipart
Pillow
scikit-learn
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset

The model was fine-tuned on a 14-class satellite dataset with the following classes:

```
Agriculture · Airport · Beach · City · Desert · Forest
Grassland · Highway · Lake · Mountain · Parking · Port · Railway · River
```

Dataset source: Roboflow — Remote Sensing Data v2

---

<p align="center">
  <b>🛰️ SkyNetra — Seeing the Earth, Intelligently.</b><br/>
</p>
