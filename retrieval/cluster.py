import numpy as np
import os
import pickle
import cv2
from sklearn.cluster import KMeans

CLUSTER_PATH = "retrieval/embeddings/kmeans_model.pkl"
N_CLUSTERS   = 10

def extract_simple_features(img_path):
    img     = cv2.imread(img_path)
    resized = cv2.resize(img, (64, 64))
    hist    = []
    for i in range(3):
        h = cv2.calcHist([resized], [i], None, [32], [0, 256])
        hist.extend(h.flatten())
    return np.array(hist, dtype="float32")

def train_clusters(image_dir: str = "data/images"):
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png"))
    ]

    if len(image_paths) == 0:
        print("❌ No images found in", image_dir)
        return None

    print(f"Extracting features from {len(image_paths)} images...")
    features = []
    for path in image_paths:
        feat = extract_simple_features(path)
        features.append(feat)

    features = np.array(features).astype("float32")

    n = min(N_CLUSTERS, len(image_paths))
    print(f"Training KMeans with {n} clusters...")
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    kmeans.fit(features)

    os.makedirs("retrieval/embeddings", exist_ok=True)
    with open(CLUSTER_PATH, "wb") as f:
        pickle.dump(kmeans, f)

    print(f"✅ Done! {n} unsupervised clusters created.")
    return kmeans

if __name__ == "__main__":
    train_clusters("data/images")