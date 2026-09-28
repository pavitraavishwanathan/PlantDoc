import os
import torch
import json
from torchvision import transforms
from PIL import Image
from model_def import MyCNNModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "..", "models", "class_names.json")) as f:
    class_names = json.load(f)

model = MyCNNModel(len(class_names))
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "..", "models", "plantdoc_image_model.pt"),
        map_location="cpu"
    )
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image: Image.Image) -> str:
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        pred = outputs.argmax(dim=1).item()
    return class_names[pred]
