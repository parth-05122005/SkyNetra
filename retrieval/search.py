import faiss
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.feature_extractor import extract_features

INDEX_PATH      = "retrieval/embeddings/index.faiss"
IMAGE_LIST_PATH = "retrieval/embeddings/image_paths.txt"

# Load index once when module is imported
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(IMAGE_LIST_PATH, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    print("✅ FAISS index loaded successfully.")
else:
    index       = None
    image_paths = []
    print("⚠️  No FAISS index found. Run faiss_index.py first.")


def retrieve_similar(image_path: str, k: int = 5) -> dict:
    if index is None:
        return {"results": [], "error": "Index not built. Run faiss_index.py first."}

    query_vector = extract_features(image_path).astype("float32").reshape(1, -1)
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "image":            image_paths[idx],
            "similarity_score": round(float(score), 4)
        })

    return {"results": results}