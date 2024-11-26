from ultralytics import YOLO

# Load a model (escaped backslashes)
model = YOLO("D:\\license-plate-yolov\\runs\\detect\\train16\\weights\\best.pt")

# Perform object detection on an image (escaped backslashes)
results = model("D:\\license-plate-yolov\\test_images\\CK_AUS_200113012101_20240619101605314_X372Y611W24H24_Van_Mitsubishi_white_051_01_03752.jpg", save=True)

# Show results
results[0].show()
