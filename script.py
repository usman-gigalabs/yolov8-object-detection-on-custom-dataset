from ultralytics import YOLO

from IPython.display import display, Image

model = YOLO('yolov8n.pt')

#source = '/home/gigalabs/Downloads/Fruit-Wallpaper-28-2560x1600-1-2048x1280.jpg'

# results = model.predict(source=source, conf=0.25, save=True)


# Train the model
results = model.train(data='datasets/data.yaml', epochs=25, imgsz=800, save=True, plots=True)