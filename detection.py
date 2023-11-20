import cv2
import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import numpy as np
from datetime import datetime
now = datetime.today().strftime("%Y/%m/%d %H:%M:%S") 

# Replace the relative path to your weight file
model_path = "weights/yolov8n.pt"

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Highway CCTV")
    st.selectbox("서울시 강남구 CCTV",("삼성동",'신사동','청담동','역삼동'))
    st.selectbox("서울시 서대문구 CCTV",('연희동','신촌동','홍제동','북아현동'))
         # Adding header to sidebar
    # Adding file uploader to sidebar for selecting videos
    source_vid = st.sidebar.selectbox(
        "Choose a video...",
        ["video.mp4"])

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
    
    st.sidebar.title('detected objects')

# Creating main page heading
st.title("Object Detection using YOLOv8")

col1,col2 = st.columns(2)

col1.metric('Time',now )
col2.metric("LEVEL", "1.2", "-0.2")

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if source_vid is not None:
    vid_cap = cv2.VideoCapture("video.mp4")
    st_frame = st.empty()

    # Get the first frame for ROI selection
    success, image = vid_cap.read()

    if success:
        # Resize the frame
        image = cv2.resize(image, (720, int(720*(9/16))))

        # Get user-selected ROI using cv2.selectROI
        roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")  # Close the window after ROI selection
        st.markdown('-----')
        st.subheader("statistics")
        (col1,col2) = st.columns(2)
        val = None
        st.markdown('----')
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()

            if success:
                # Resize the frame
                image = cv2.resize(image, (720, int(720*(9/16))))

                # Draw the ROI on the frame
                x, y, w, h = roi
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the frame based on the selected ROI
                roi_image = image[y:y+h, x:x+w]

                # Perform object detection on the cropped frame
                res = model.predict(roi_image, conf=confidence)
                name = model.names
                res_plotted = res[0].plot()
                

                for label in res:
                    for car in label.boxes.cls:
                        class_name = name[int(car)]
                        st.sidebar.write(class_name)
                        
                # Merge the annotated ROI frame with the original frame
                annotated_frame = image.copy()
                annotated_frame[y:y+h, x:x+w] = res_plotted

                # Display the annotated frame
                st_frame.image(annotated_frame,
                               caption='Detected Video with ROI',
                               channels="BGR",
                               use_column_width=True
                               )

            else:
                vid_cap.release()
                break

    