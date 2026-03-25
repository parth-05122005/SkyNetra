from ultralytics import YOLO
import cv2
import os
import json

model = None

def load_model():
    global model
    if model is None:
        model = YOLO("weights/best (2).pt")
    return model

def run_detection(image_path: str) -> dict:
    m = load_model()
    results = m.predict(source=image_path, conf=0.6, save=False, verbose=False)

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = m.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "bbox": [x1, y1, x2, y2]
        })
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs("outputs", exist_ok=True)
    annotated_path = "outputs/detected.jpg"
    cv2.imwrite(annotated_path, img)

    label_path = "outputs/sample.txt"
    with open(label_path, "w") as f:
        for det in detections:
            names_list = list(m.names.values())
            ci = names_list.index(det["label"]) if det["label"] in names_list else 0
            x1, y1, x2, y2 = det["bbox"]
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            f.write(f"{ci} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    return {
        "objects": detections,
        "annotated_image": annotated_path,
        "label_file": label_path
    }

# Also add this so Member 3's backend can call detect_objects() too
def detect_objects(image_path: str) -> dict:
    return run_detection(image_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>")
        sys.exit(1)
    output = run_detection(sys.argv[1])
    print(json.dumps(output, indent=2))