import streamlit as st
import requests
from PIL import Image
import io

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyNetra",
    page_icon="🛰️",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; font-size:2.8rem;'>🛰️ SkyNetra</h1>
    <p style='text-align:center; color:gray; font-size:1.1rem;'>
        Satellite Visual Search, Detection & Auto-Labelling System
    </p>
    <hr/>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.title("About SkyNetra")
    st.markdown("""
    **SkyNetra** is an AI-powered pipeline for satellite imagery.

    **Features:**
    - Object detection (YOLOv8)
    - Auto label generation (YOLO format)
    - Visual similarity search (FAISS)

    **Tech Stack:**
    - YOLOv8 · ResNet50 · FAISS
    - FastAPI · Streamlit

    ---
    **Team:**
    - Member 1: Detection
    - Member 2: Retrieval
    - Member 3: Integration
    """)

# ── Upload Section ───────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Upload a Satellite Image",
        type=["jpg", "jpeg", "png"],
        help="Upload any satellite image to detect objects and find similar images."
    )

if uploaded_file is not None:
    # Show uploaded image
    st.subheader("📸 Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Your satellite image", width="stretch")

    # Analyze button
    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):

        with st.spinner("🤖 Running AI analysis... please wait"):
            try:
                uploaded_file.seek(0)
                response = requests.post(
                    "http://localhost:8000/analyze",
                    files={"file": (uploaded_file.name, uploaded_file, "image/jpeg")},
                    timeout=60
                )
                result = response.json()

            except Exception as e:
                st.error(f"❌ Could not connect to backend: {e}")
                st.stop()

        st.success("✅ Analysis complete!")
        st.divider()

        # ── Two column results ─────────────────────────────────
        left, right = st.columns(2)

        # LEFT: Detection Results
        with left:
            st.subheader("📦 Detected Objects")
            objects = result["detection"].get("objects", [])

            # ── Learning type summary ──────────────────────────────────
            summary = result["detection"].get("summary", {})
            if summary:
                st.divider()
                st.subheader("🧠 Learning Mode Breakdown")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Objects",      summary.get("total_objects", 0))
                c2.metric("✅ Supervised",      summary.get("supervised_count", 0),
                        help="Detected with confidence > 0.6 using YOLOv8")
                c3.metric("🔵 Unsupervised",    summary.get("unsupervised_count", 0),
                        help="Unknown objects grouped by K-Means clustering")

            # ── Per object learning type badge ────────────────────────
            for obj in objects:
                mode  = obj.get("learning_type", "supervised")
                color = "🟢" if mode == "supervised" else "🔵"
                badge = "Supervised" if mode == "supervised" else f"Unsupervised (Cluster {obj.get('cluster_id', '?')})"
                conf  = obj["confidence"]
                st.markdown(
                    f"{color} **{obj['label'].upper()}** — "
                    f"`{badge}` — "
                    f"Confidence: `{conf:.0%}` — "
                    f"BBox: `{obj['bbox']}`"
                )

            if objects:
                for obj in objects:
                    conf = obj["confidence"]
                    color = "🟢" if conf > 0.85 else "🟡" if conf > 0.70 else "🔴"
                    st.markdown(
                        f"{color} **{obj['label'].upper()}** — "
                        f"Confidence: `{conf:.0%}` — "
                        f"BBox: `{obj['bbox']}`"
                    )
            else:
                st.info("No objects detected.")

            # Show annotated image if available
            annotated = result["detection"].get("annotated_image")
            if annotated and annotated != "outputs/detected.jpg":
                st.image(
                    f"http://localhost:8000/{annotated}",
                    caption="Annotated image with bounding boxes"
                )

        # RIGHT: Retrieval Results
        with right:
            st.subheader("🔁 Similar Satellite Images")
            similar = result["retrieval"].get("results", [])

            if similar:
                for i, match in enumerate(similar):
                    score = match["similarity_score"]
                    st.markdown(
                        f"**#{i+1}** `{match['image'].split('/')[-1]}` — "
                        f"Similarity: `{score:.0%}`"
                    )
                    # Show image if it exists locally
                    try:
                        img = Image.open(match["image"])
                        st.image(img, width=200)
                    except:
                        st.caption("_(image preview not available)_")
            else:
                st.info("No similar images found.")

        st.divider()

        # ── Download label file ────────────────────────────────
        label_file = result["detection"].get("label_file", "")
        if label_file:
            try:
                with open(label_file, "r") as f:
                    label_content = f.read()
                st.download_button(
                    label="⬇️ Download YOLO Label File (.txt)",
                    data=label_content,
                    file_name="skynetra_labels.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except:
                st.caption("Label file will be available after real detection runs.")

else:
    # Placeholder when no image is uploaded
    st.info("👆 Upload a satellite image above to get started.")
    st.markdown("""
    **What SkyNetra will detect:**
    - 🏗️ Buildings & structures
    - 🛣️ Roads & highways
    - 🚗 Vehicles
    - ✈️ Aircraft & airports
    - 🚢 Ships & harbors
    """)