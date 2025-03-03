import zipfile
import shutil
import os
from ultralytics import YOLO
import torch

# Check if dataset.zip exists
zip_path = "/content/dataset.zip"
if zipfile.is_zipfile(zip_path):
    print("✅ The file is a valid ZIP archive.")
    !unzip /content/dataset.zip -d /content/dataset/
else:
    print("❌ The file is not a valid ZIP archive. Please re-upload it.")

# Move dataset folders
shutil.move("/content/dataset/dataset/train", "/content/dataset/")
shutil.move("/content/dataset/dataset/valid", "/content/dataset/")
shutil.move("/content/dataset/dataset/test", "/content/dataset/")
shutil.rmtree("/content/dataset/dataset", ignore_errors=True)

print("File structure corrected ✅")

# Check if data.yaml exists, otherwise create it
yaml_path = "/content/dataset/data.yaml"
if not os.path.exists(yaml_path):
    print("data.yaml not found ❌, creating...")
    yaml_content = """
    train: /content/dataset/train/images
    val: /content/dataset/valid/images
    test: /content/dataset/test/images

    nc: 6
    names: ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'truck']
    """
    with open(yaml_path, "w") as file:
        file.write(yaml_content)
    print("data.yaml created ✅")

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Load YOLO model
model = YOLO("yolov8n.pt")

# Train YOLOv8 model
model.train(data=yaml_path, epochs=50, imgsz=640, batch=16)

print("✅ Training complete! Model saved in 'runs/detect/train/weights/best.pt'")