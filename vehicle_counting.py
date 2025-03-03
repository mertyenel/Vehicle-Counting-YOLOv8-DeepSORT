# Install YOLOv8 (Ultralytics)
!pip install ultralytics

# Install DeepSORT for object tracking
!pip install deep_sort_realtime

import os

# Paths for the trained model and test video
model_path = "/content/best.pt"
video_path = "/content/test_video.mp4"

# Check if the trained model file exists
if os.path.exists(model_path):
    print("✅ Model file found:", model_path)
else:
    print("❌ Model file not found! Please upload it again.")

# Check if the test video file exists
if os.path.exists(video_path):
    print("✅ Test video found:", video_path)
else:
    print("❌ Test video not found! Please upload it again.")

!pip install ultralytics
from ultralytics import YOLO
import torch

# Load the trained YOLOv8 model
model = YOLO("best.pt")

# Run inference on a sample image
results = model("/content/sample.jpeg")

# Display the results
results[0].show()

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab.patches import cv2_imshow

# Load the YOLO model
model = YOLO("/content/best.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=15,
    n_init=1,
    max_cosine_distance=0.2,
    nn_budget=100
)

# Open test video
video_path = "/content/test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the counting line position (e.g., 60% of the video height)
line_position = int(frame_height * 0.6)

# Define the output video path
output_path = "/content/test_video_counted.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize vehicle count and tracking variables
vehicle_count = 0
counted_track_ids = set()
previous_positions = {}

# Define class names for the dataset
class_names = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'truck']

# Define confidence and IOU thresholds
conf_threshold = 0.5 
iou_threshold = 0.4

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1) YOLO object detection
    results = model(frame, iou=iou_threshold)

    detections = []
    if len(results) > 0:
        boxes_data = results[0].boxes
        for box in boxes_data:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            # Ensure the bounding box stays within frame limits
            x1 = np.clip(x1, 0, frame_width)
            y1 = np.clip(y1, 0, frame_height)
            x2 = np.clip(x2, 0, frame_width)
            y2 = np.clip(y2, 0, frame_height)

            conf = float(box.conf[0].cpu().numpy())
            if conf < conf_threshold:
                continue  # Skip low-confidence detections

            cls_id = int(box.cls[0].cpu().numpy())

            # Convert XYXY format to XYWH format
            w = x2 - x1
            h = y2 - y1

            # If class name is not in the list, assign as "vehicle"
            if 0 <= cls_id < len(class_names):
                label = class_names[cls_id]
            else:
                label = "vehicle"

            # Add to DeepSORT tracker
            detections.append(((x1, y1, w, h), conf, cls_id))

    # 2) Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw the red counting line
    cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 3)

    # 3) Process each confirmed track for counting
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # [left, top, right, bottom]
        x1, y1, x2, y2 = map(int, ltrb)
        center_y = (y1 + y2) // 2

        det_class = track.det_class if hasattr(track, 'det_class') else None
        if det_class is not None and 0 <= det_class < len(class_names):
            label = class_names[det_class]
        else:
            label = "vehicle"

        # Check if the vehicle has crossed the counting line
        prev_y = previous_positions.get(track_id, None)
        if prev_y is not None:
            if prev_y < line_position <= center_y and track_id not in counted_track_ids:
                vehicle_count += 1
                counted_track_ids.add(track_id)

        previous_positions[track_id] = center_y

        # Draw bounding box around detected vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display total vehicle count on the frame
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Save the processed frame to the output video
    out.write(frame)

    # Display the frame (for Google Colab)
    cv2_imshow(frame)

# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Processing complete. Video saved. Total {vehicle_count} vehicles passed.")