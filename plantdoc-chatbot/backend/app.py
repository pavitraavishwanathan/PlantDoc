import io
import os
import json
import torch

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model_def import MyCNNModel

# =========================================================
# App setup
# =========================================================
app = FastAPI(title="PlantDoc – Image + Text Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# Load IMAGE class names (ORDER MATTERS)
# =========================================================
with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    CLASS_NAMES = json.load(f)

print("✅ Image classes:", len(CLASS_NAMES))  # should be 38

# =========================================================
# Load IMAGE model
# =========================================================
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "plantdoc_image_model.pt")

image_model = MyCNNModel(num_classes=len(CLASS_NAMES))
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
image_model.eval()

# Image preprocessing (must match training)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image: Image.Image):
    image = image_transform(image)
    return image.unsqueeze(0)

# =========================================================
# Load TEXT model (NEW TRAINED TRANSFORMER)
# =========================================================
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "text_model")

text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
text_model.eval()

print("✅ Text classes:", text_model.config.num_labels)

# =========================================================
# IMAGE prediction endpoint
# =========================================================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = image_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    class_index = predicted.item()
    disease_name = CLASS_NAMES[class_index]

    return {
        "mode": "image",
        "disease": disease_name,
        "confidence": round(confidence.item(), 4)
    }

# =========================================================
# TEXT prediction endpoint (FINAL, CORRECT)
# =========================================================
@app.post("/text_predict")
async def predict_text(text: str = Form(...)):
    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = probs.argmax().item()

    return {
        "mode": "text",
        "disease": text_model.config.id2label[pred_id],
        "confidence": round(probs[pred_id].item(), 4)
    }

# =========================================================
# Health check
# =========================================================
@app.get("/")
def root():
    return {"status": "PlantDoc backend running 🚀"}

@app.post("/chat")
async def chat(
    text: str = Form(None),
    image: UploadFile = File(None)
):
    # Case 1: both text + image
    if text and image:
        # --- IMAGE ---
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = preprocess_image(img)

        with torch.no_grad():
            outputs = image_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            img_conf, img_pred = torch.max(probs, dim=1)

        image_disease = CLASS_NAMES[img_pred.item()]

        # --- TEXT ---
        inputs = text_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        )

        with torch.no_grad():
            txt_outputs = text_model(**inputs)
            txt_probs = torch.softmax(txt_outputs.logits, dim=1)[0]
            txt_pred = txt_probs.argmax().item()

        text_disease = text_model.config.id2label[txt_pred]

        return {
            "mode": "image + text",
            "final_disease": image_disease,
            "image_confidence": round(img_conf.item(), 4),
            "text_hint": text_disease
        }

    # Case 2: image only
    if image:
        return await predict_image(image)

    # Case 3: text only
    if text:
        return await predict_text(text)

    return {"message": "No input provided"}
