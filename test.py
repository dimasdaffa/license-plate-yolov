from ultralytics import YOLO

# Load a model (escaped backslashes)
model = YOLO("D:\\license-plate-yolov\\runs\\detect\\train16\\weights\\best.pt")

# Perform object detection on an image (escaped backslashes)
results = model("D:\license-plate-yolov\\test_images\plat (1).jpg", save=True)

# Show results
results[0].show()
