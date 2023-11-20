import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

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

#upload file
video_path = st.sidebar.file_uploader("Choose a video file", type=["mp4"])

# Allow the user to select a ROI with the mouse
if video_path is not None:
    # Convert video bytes to OpenCV format
    video_nparray = np.frombuffer(video_path.read(), dtype=np.uint8)
    cap = cv2.VideoCapture(video_nparray)

    # Write the video bytes to a temporary file
    temp_file_path = "temp_video.mp4"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(video_nparray)

    # Open the temporary video file with cv2.VideoCapture
    cap.open(temp_file_path)

    success, frame = cap.read()
    if success:
        roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)

        # Start inference when the user presses Enter
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Check if Enter key is pressed
                break  # Exit the loop to start YOLOv8 inference

# Close the video capture object
        cap.release()

# Perform YOLOv8 inference and visualization
        annotated_frame = yolo_inference(frame, roi)

# Display the annotated frame
        st.image(annotated_frame)
