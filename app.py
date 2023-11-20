import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Define the region of interest (ROI)
roi_x1 = 64
roi_y1 = 273
roi_x2 = 297
roi_y2 = 690

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Crop the ROI from the frame
        roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Run YOLOv8 inference on the ROI
        results = model(roi_frame)

        # Visualize the results on the ROI
        annotated_roi_frame = results[0].plot()

        # Merge the annotated ROI frame with the original frame
        annotated_frame = frame.copy()
        annotated_frame[roi_y1:roi_y2, roi_x1:roi_x2] = annotated_roi_frame

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()