# Import the necessary libraries
from ultralytics import YOLO

# Load a YOLOv8 model (use 'yolov8n.pt' for Nano, 'yolov8s.pt' for Small, 'yolov8m.pt' for Medium, 'yolov8l.pt' for Large, etc.)
model = YOLO('yolov8s.pt')  # Using YOLOv8 small model for faster training; change as needed

# Train the model
model.train(
    data='path/to/your/dataset.yaml',  # path to your dataset.yaml file
    epochs=100,                        # number of epochs
    imgsz=640,                         # image size for training
    batch=16,                          # batch size
    device=0,                          # GPU device (use 'cpu' if no GPU)
    name='yolov8_traffic_accidents',    # save model with this name
    patience=10,                       # early stopping patience
    optimizer='Adam',                   # optimizer (can be 'Adam' or 'SGD')
    lr0=0.001,                         # initial learning rate
    pretrained=True                    # use pretrained weights
)

# Once training is done, you can test the model using:
results = model.val()  # Evaluate model performance on the validation set

# Optionally, predict on new images:
results = model.predict('path/to/your/test_image.jpg', save=True)  # save=True saves the image with predictions
