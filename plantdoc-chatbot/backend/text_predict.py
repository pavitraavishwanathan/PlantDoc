import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "text_model")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# -----------------------------
# Load label encoder
# -----------------------------
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# -----------------------------
# Load tokenizer + model (LOCAL ONLY)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.eval()

# -----------------------------
# Prediction function
# -----------------------------
def predict_text(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()

    return label_encoder.inverse_transform([pred_id])[0]
