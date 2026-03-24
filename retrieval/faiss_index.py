import faiss
import numpy as np
import os
import sys

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add to path for imports
sys.path.append(BASE_DIR)

from retrieval.feature_extractor import extract_features


def build_index():
    image_folder = os.path.join(BASE_DIR, "data", "images")

    image_paths = []
    vectors = []

    print(f"📂 Scanning folder: {image_folder}")

    if not os.path.exists(image_folder):
        print("❌ Folder does not exist:", image_folder)
        return

    for fname in os.listdir(image_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(image_folder, fname)
            try:
                vec = extract_features(full_path)
                vectors.append(vec)
                image_paths.append(full_path)
                print(f"  ✅ Processed: {fname}")
            except Exception as e:
                print(f"  ❌ Skipped {fname}: {e}")

    if not vectors:
        print("❌ No images found! Add images to data/images/")
        return

    matrix = np.stack(vectors).astype("float32")

    index = faiss.IndexFlatIP(2048)
    index.add(matrix)

    # Save paths properly
    save_dir = os.path.join(BASE_DIR, "retrieval", "embeddings")
    os.makedirs(save_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))

    with open(os.path.join(save_dir, "image_paths.txt"), "w") as f:
        f.write("\n".join(image_paths))

    print(f"\n✅ Index built successfully with {len(image_paths)} images.")
    print("📁 Saved to retrieval/embeddings/")


if __name__ == "__main__":
    build_index()