import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from src.vision_engine.predict import DiseaseClassifierEngine

# Page Configuration
st.set_page_config(
    page_title="Smart Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Smart Crop Disease Detection & Drone Surveillance")
st.markdown("Upload a crop leaf image or a drone video stream to detect disease health status in real time.")

@st.cache_resource
def load_engine():
    try:
        # Require at least 65% confidence before declaring a disease
        return DiseaseClassifierEngine(conf_threshold=0.65)
    except Exception as e:
        st.error(f"Error loading model weights from predict.py: {e}")
        return None

engine = load_engine()

# Sidebar Navigation
st.sidebar.title("🕹️ Control Panel")
app_mode = st.sidebar.selectbox("Select Mode", ["Image Analysis", "Drone Video Feed"])

if app_mode == "Image Analysis":
    st.subheader("📷 Image Diagnostics")
    uploaded_file = st.file_uploader("Upload Crop Leaf Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and engine:
        col1, col2 = st.columns([1, 1])
        
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        # Convert RGB (PIL) to BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        with col1:
            st.markdown("### 🖼️ Input Image")
            st.image(image, caption="Uploaded Leaf Sample", use_container_width=True)
            
        with col2:
            st.markdown("### 🔬 Diagnostic Actions")
            st.write("Click below to run the AI disease identification pipeline.")
            
            if st.button("🔍 Run Disease Diagnostics", use_container_width=True):
                with st.spinner("Applying HSV masking & analyzing leaf features..."):
                    res = engine.predict(img_bgr)
                
                st.divider()
                if res["status"] == "UNCERTAIN":
                    st.warning(
                        f"⚠️ **Uncertain Diagnosis ({res['confidence']}% confidence)**\n\n"
                        f"{res['message']}\n\n"
                        f"*Closest match was: {res['raw_prediction']}*"
                    )
                elif "healthy" in res["raw_class"].lower():
                    st.success(
                        f"✅ **Status: Healthy**\n\n"
                        f"**Diagnosis:** {res['crop_disease']}\n\n"
                        f"**Confidence:** {res['confidence']}%"
                    )
                else:
                    st.error(
                        f"🚨 **Status: Disease Detected**\n\n"
                        f"**Diagnosis:** {res['crop_disease']}\n\n"
                        f"**Confidence:** {res['confidence']}%"
                    )

elif app_mode == "Drone Video Feed":
    st.subheader("🛸 Drone Vision Analytics")
    video_file = st.file_uploader("Upload Drone Flight Video", type=["mp4", "avi", "mov"])
    
    if video_file and engine:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        st.info("▶️ Processing drone flight video stream...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            res = engine.predict(frame)
            
            if res["status"] == "SUCCESS":
                label = f"{res['crop_disease']} ({res['confidence']}%)"
                color = (0, 255, 0) if "healthy" in res["raw_class"].lower() else (0, 0, 255)
            else:
                label = f"UNCERTAIN ({res['confidence']}%)"
                color = (0, 165, 255) # Orange
                
            # Overlay HUD telemetry
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
        cap.release()