import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set Streamlit page config
st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="wide"
)

# Title & Description
st.title("🌿 Field-Optimized Plant Disease Detector")
st.markdown("""
Upload a field photograph of a leaf or crop to detect diseases in real-time using our **YOLOv8 Field-Optimized Model**.
""")

# Sidebar settings
st.sidebar.header("⚙️ Model Settings")

# Model Loading Function with caching to avoid reloading on every click
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    # Updated path to match repository structure
    model = load_model("models/disease_classifier_v2/weights/best.pt")
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model file: {e}")
    st.stop()

# Confidence threshold slider (as recommended in our domain-gap audit)
conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help="Increase threshold to filter out background clutter false positives in messy field photos."
)

# File Uploader
uploaded_file = st.file_uploader(
    "Choose a leaf image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display Original Image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_container_width=True)
    
    # Run Inference button
    with col2:
        st.subheader("🔍 Disease Detection Output")
        
        with st.spinner("Analyzing leaf pathology..."):
            # Run YOLOv8 inference
            results = model.predict(source=image, conf=conf_threshold)
            
            # Extract plotted result image
            res_plotted = results[0].plot()
            
            # Display detection visualization
            st.image(res_plotted, channels="BGR", use_container_width=True)
    
    # Detailed Detections Breakdown
    st.markdown("---")
    st.subheader("📊 Detection Summary")
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        detected_classes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detected_classes.append((cls_name, conf))
        
        # Display table of results
        st.markdown(f"**Total Diseases / Objects Detected:** `{len(detected_classes)}`")
        
        for name, conf in detected_classes:
            st.warning(f"🚨 **Detected:** `{name}` (Confidence: `{conf * 100:.1f}%`)")
            
    else:
        st.info("🟢 No disease symptoms detected above the current confidence threshold.")