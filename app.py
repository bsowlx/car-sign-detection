import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# Page setup
st.set_page_config(page_title="Traffic Sign Detection", page_icon="🛑", layout="wide")
st.title("🛑 Traffic Sign Detection using YOLOv11n")

# Sidebar settings
st.sidebar.header("Settings")
conf_thresh = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.05)

# Load model
@st.cache_resource
def get_model():
    return YOLO("best.onnx")

try:
    model = get_model()
    st.sidebar.success("Model loaded!")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Input selection
option = st.sidebar.radio("Source", ["Upload", "Sample"])
file = None
is_video = False
video_path = None

if option == "Upload":
    file = st.file_uploader("Upload image or video", type=["jpg", "png", "mp4"])
    if file:
        if file.type.startswith('video'):
            is_video = True
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            video_path = tfile.name
        else:
            file = Image.open(file)

else:
    samples = {
        "Image 1": "car/test/images/road104_png.rf.4468adcd7c2dcfc04377146de785c0a6.jpg",
        "Image 2": "car/test/images/road112_png.rf.b765c2b5d3a2145d317b0ceadfb08464.jpg",
        "Video": "video.mp4"
    }
    choice = st.selectbox("Select sample", list(samples.keys()))
    path = samples[choice]
    
    if os.path.exists(path):
        if path.endswith(".mp4"):
            is_video = True
            video_path = path
            file = open(path, "rb")
        else:
            file = Image.open(path)
    else:
        st.error("Sample file not found.")

# Processing
if file:
    if not is_video:
        st.image(file, caption="Input", width="stretch")
        
        if st.button("Detect"):
            # Run detection
            results = model.predict(np.array(file), conf=conf_thresh)
            
            # Show result
            res_img = results[0].plot()
            st.image(res_img, caption="Result", width="stretch")
            
            # List detections
            st.subheader("Found:")
            for box in results[0].boxes:
                name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                st.write(f"- {name}: {conf:.2f}")
                
    else:
        st.video(file)
        
        if st.button("Detect in Video"):
            st_frame = st.empty()
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Detect and plot
                res = model.predict(frame, conf=conf_thresh)
                res_frame = res[0].plot()
                
                # Show frame
                st_frame.image(cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB), width="stretch")
            
            cap.release()

st.markdown("---")
st.write("Simple YOLOv11n App")
