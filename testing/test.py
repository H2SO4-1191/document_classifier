from ultralytics import YOLO

model = YOLO(r"./models/document_classifier_v3.pt") # Path to where the 'best.pt' model of the latest train exists (check: './runs/classify/train/weights/best.pt')

img = r"C:/Users/H2SO4/OneDrive/Documents/Personal Documents/residance_back.jpg" # Path to an image to predict

results = model(img)
print(results)
