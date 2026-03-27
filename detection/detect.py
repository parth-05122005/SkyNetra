from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
import pickle
from sklearn.cluster import KMeans

model = None

# ── Load KMeans if available ──────────────────────────────────
CLUSTER_PATH = "retrieval/embeddings/kmeans_model.pkl"

def load_kmeans():
    if os.path.exists(CLUSTER_PATH):
        with open(CLUSTER_PATH, "rb") as f:
            return pickle.load(f)
    return None

def load_model():
    global model
    if model is None:
        model = YOLO("weights/best.pt")
    return model

# ── Crop detected region for clustering ──────────────────────
def crop_region(img, bbox):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img
    return crop

# ── Simple color histogram feature for unknown objects ───────
def extract_simple_features(img_crop):
    resized = cv2.resize(img_crop, (64, 64))
    hist = []
    for i in range(3):
        h = cv2.calcHist([resized], [i], None, [32], [0, 256])
        hist.extend(h.flatten())
    return np.array(hist, dtype="float32")

# ── Main detection function ───────────────────────────────────
def run_detection(image_path: str) -> dict:
    m = load_model()
    kmeans = load_kmeans()

    results = m.predict(
        source=image_path,
        conf=0.3,        # lower threshold to catch uncertain detections
        save=False,
        verbose=False
    )

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    detections = []

    for box in results[0].boxes:
        cls_id     = int(box.cls[0])
        label      = m.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── DECISION: Supervised or Unsupervised ─────────────
        if confidence >= 0.6:
            # ✅ SUPERVISED — confident prediction, use label
            learning_type = "supervised"
            final_label   = label
            cluster_id    = None

        else:
            # ❓ UNSUPERVISED — low confidence, use clustering
            learning_type = "unsupervised"
            crop = crop_region(img, [x1, y1, x2, y2])
            feat = extract_simple_features(crop).reshape(1, -1)

            if kmeans is not None:
                cluster_id  = int(kmeans.predict(feat)[0])
                final_label = f"Unknown Pattern (Cluster {cluster_id})"
            else:
                cluster_id  = None
                final_label = f"Unknown Object ({label}?)"

        # ── Draw bounding box ─────────────────────────────────
        color = (0, 255, 0) if learning_type == "supervised" else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f'{final_label} {confidence:.2f}',
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
        )

        detections.append({
            "label":         final_label,
            "confidence":    round(confidence, 2),
            "bbox":          [x1, y1, x2, y2],
            "learning_type": learning_type,
            "cluster_id":    cluster_id
        })

    # ── Save annotated image ──────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    annotated_path = "outputs/detected.jpg"
    cv2.imwrite(annotated_path, img)

    # ── Save YOLO label file ──────────────────────────────────
    label_path = "outputs/sample.txt"
    with open(label_path, "w") as f:
        for det in detections:
            names_list = list(m.names.values())
            raw_label  = det["label"].split(" ")[0]
            ci = names_list.index(raw_label) if raw_label in names_list else 0
            x1, y1, x2, y2 = det["bbox"]
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            f.write(f"{ci} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # ── Summary stats ─────────────────────────────────────────
    supervised_count   = sum(1 for d in detections if d["learning_type"] == "supervised")
    unsupervised_count = sum(1 for d in detections if d["learning_type"] == "unsupervised")

    return {
        "objects":          detections,
        "annotated_image":  annotated_path,
        "label_file":       label_path,
        "summary": {
            "total_objects":       len(detections),
            "supervised_count":    supervised_count,
            "unsupervised_count":  unsupervised_count
        }
    }

def detect_objects(image_path: str) -> dict:
    return run_detection(image_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>")
        sys.exit(1)
    output = run_detection(sys.argv[1])
    print(json.dumps(output, indent=2))