import os
import cv2
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image

# Define dataset classes
classes = ['__background__', 'Apple', 'Banana', 'Orange']
num_classes = len(classes)

# Safe model loading for Jetson (from GPU 1 to current device)
def load_trained_model(model_path, num_classes, device):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Inference function
def predict(image, model, device):
    image = [image.to(device)]
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Visualization with bounding boxes
def visualize_prediction(image, prediction, threshold=0.4):
    from torchvision.utils import draw_bounding_boxes
    import matplotlib.pyplot as plt

    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    class_names = [classes[i] for i in labels]
    text = [f"{name}: {score:.2f}" for name, score in zip(class_names, scores)]
    drawn_image = draw_bounding_boxes(image.mul(255).byte(), boxes, text, width=4)

    return drawn_image

# Image preprocessing
def preprocess_image(image_pil):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image_pil)

# Setup Jetson-compatible device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define model path
model_path = os.path.expanduser('fruit.pth')  # Ensure this exists
trained_model = load_trained_model(model_path, num_classes, device)

# Start video capture
#cap = cv2.VideoCapture(0)  # Change to your camera index or RTSP stream
cap = cv2.VideoCapture('/dev/video0')
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR to RGB and to PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    image_tensor = preprocess_image(image_pil).to(device)

    # Predict
    prediction = predict(image_tensor, trained_model, device)

    # Visualize
    drawn = visualize_prediction(image_tensor.cpu(), prediction)
    drawn_np = drawn.permute(1, 2, 0).byte().numpy()
    drawn_bgr = cv2.cvtColor(drawn_np, cv2.COLOR_RGB2BGR)

    cv2.imshow("Real-Time Detection", drawn_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
