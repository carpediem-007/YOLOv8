from ultralytics import YOLO
import torch
import wget
from PIL import Image
import numpy as np
import time
import cv2
import streamlit as st

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
# model.val()

def yolo_inference(frame, roi):
    """Performs YOLOv8 inference and visualization on the given frame and ROI.

    Args:
        frame: A NumPy array representing the video frame.
        roi: A NumPy array representing the region of interest in the frame.

    Returns:
        A NumPy array representing the annotated frame.
    """

    # Crop the ROI from the frame
    roi_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

    # Run YOLOv8 inference on the ROI
    results = model(roi_frame)

    # Visualize the results on the ROI
    annotated_roi_frame = results[0].plot()

    # Merge the annotated ROI frame with the original frame
    annotated_frame = frame.copy()
    annotated_frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = annotated_roi_frame

    # Draw a rectangle around the ROI
    color = (0, 255, 0)
    cv2.rectangle(annotated_frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), color, 2)

    return annotated_frame

st.title("YOLOv8 Inference with Streamlit")
st.sidebar.button("서울시 종로구 CCTV")
st.sidebar.button("서울시 강남구 CCTV")
st.sidebar.button("서울시 서대문구 CCTV")


vid_file = None
vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
if vid_bytes:
    vid_file = "output.mp4"
    with open(vid_file, 'wb') as out:
        out.write(vid_bytes.read())

cap = cv2.VideoCapture(vid_file)

# count=0
output = st.empty()
while True:
    ret,frame=cap.read()
    if not ret:
        break
    # count += 1
    # if count % 3 != 0:
    #     continue

    frame=cv2.resize(frame,(1020,500))

#upload file
#video_path = st.sidebar.file_uploader("Choose a video file", type=["mp4"])

#if ret:
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)

    # Start inference when the user presses Enter
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Check if Enter key is pressed
            break  # Exit the loop to start YOLOv8 inference

    annotated_frame = yolo_inference(frame, roi)

# Close the video capture object
    cap.release()

# Perform YOLOv8 inference and visualization
        

# Display the annotated frame
    st.image(annotated_frame)
