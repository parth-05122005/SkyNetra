def detect_objects(image_path: str) -> dict:
    # MOCK — replace with Member 1's real code later
    return {
        "objects": [
            {"label": "building", "confidence": 0.91, "bbox": [120, 80, 340, 290]},
            {"label": "road",     "confidence": 0.87, "bbox": [0,  310, 640, 380]},
            {"label": "vehicle",  "confidence": 0.72, "bbox": [200, 150, 260, 180]}
        ],
        "annotated_image": "outputs/detected.jpg",
        "label_file":      "outputs/sample.txt"
    }