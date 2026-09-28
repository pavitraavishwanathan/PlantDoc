# PlantDoc AI

PlantDoc AI is an AI-powered plant disease detection and assistance system that combines **image-based plant disease classification** with **text-based disease prediction**.

The project integrates deep learning, natural language processing, a FastAPI backend, and a web-based frontend to provide an AI-assisted plant health application.

---

## Features

*  Plant disease detection from leaf images
*  Text-based plant disease prediction
*  Combined image + text prediction
*  Prediction confidence scores
*  Custom CNN for image classification
*  Transformer-based text classification
*  FastAPI backend
*  HTML, CSS, and JavaScript frontend
*  PlantVillage dataset for image-based training

---

##  System Architecture

```text
                         PlantDoc AI
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Image Prediction          Text Prediction
                 |                         |
                 v                         v
            Custom CNN              Transformer Model
                 |                         |
                 +------------+------------+
                              |
                              v
                       FastAPI Backend
                              |
                              v
                       Web Frontend
```

---

##  Project Structure

```text
Plantdoc/
│
├── plantdoc-chatbot/
│   │
│   ├── backend/
│   │   ├── app.py
│   │   ├── chat_logic.py
│   │   ├── extract_text_labels.py
│   │   ├── image_predict.py
│   │   ├── model_def.py
│   │   ├── text_predict.py
│   │   └── requirements.txt
│   │
│   └── frontend/
│       ├── index.html
│       ├── script.js
│       └── style.css
│
├── training/
│   └── train_model.py
│
├── .gitignore
└── README.md
```

---

##  AI Components

### 1. Image-Based Disease Detection

PlantDoc uses a custom convolutional neural network defined in `model_def.py` for image-based plant disease classification.

The image processing pipeline is:

```text
Plant Leaf Image
       ↓
Resize to 224 × 224
       ↓
Tensor Conversion
       ↓
Image Normalization
       ↓
Custom CNN
       ↓
Disease Classification
       ↓
Confidence Score
```

The CNN contains:

* Convolutional layers
* ReLU activation
* Max pooling
* Adaptive average pooling
* Fully connected layers
* Dropout

The relevant implementation is located in:

```text
plantdoc-chatbot/backend/model_def.py
plantdoc-chatbot/backend/image_predict.py
```

---

### 2. Text-Based Disease Prediction

The text component uses a **Hugging Face Transformer sequence-classification model** to process plant/disease-related text input.

The workflow is:

```text
User Query
    ↓
Tokenization
    ↓
Transformer Model
    ↓
Class Probabilities
    ↓
Predicted Disease
    ↓
Confidence Score
```

The relevant implementation is located in:

```text
plantdoc-chatbot/backend/text_predict.py
plantdoc-chatbot/backend/chat_logic.py
```

---

### 3. Combined Image + Text Prediction

The `/chat` endpoint supports three input modes:

* Image + text
* Image only
* Text only

When both an image and text are provided, the system performs predictions using both models and returns the image prediction together with the text-based disease prediction.

---

##  Backend API

The application uses **FastAPI** for the backend.

### Health Check

```text
GET /
```

Returns the current backend status.

### Image Prediction

```text
POST /predict
```

Accepts a plant leaf image and returns:

* Predicted disease
* Prediction confidence
* Prediction mode

### Text Prediction

```text
POST /text_predict
```

Accepts text input and returns:

* Predicted disease
* Prediction confidence
* Prediction mode

### Combined Chat

```text
POST /chat
```

Supports:

```text
Image + Text
Image only
Text only
```

---

##  Dataset

The image classification component uses the **PlantVillage dataset**.

The dataset is intentionally excluded from this repository because of its size.

The local dataset directory used by the training script is:

```text
dataset/PlantVillage
```

---

##  Model Training

The training code is located in:

```text
training/train_model.py
```

The training script uses:

* PyTorch
* Torchvision
* `ImageFolder`
* ResNet18
* Cross-Entropy Loss
* Adam optimizer

The general training workflow is:

```text
PlantVillage Dataset
        ↓
ImageFolder
        ↓
Image Preprocessing
        ↓
ResNet18
        ↓
Model Training
        ↓
Saved Model
```

> **Note:** The current inference code uses the custom `MyCNNModel` architecture, while the current training script contains a ResNet18-based training workflow. These two components should be aligned before using the training script to regenerate the exact image model used by the application.

---

## Technologies Used

### Programming Languages

* Python
* HTML
* CSS
* JavaScript

### Machine Learning & AI

* PyTorch
* Torchvision
* Hugging Face Transformers
* Scikit-learn
* NumPy
* Pillow

### Backend

* FastAPI
* Uvicorn

### Frontend

* HTML5
* CSS3
* JavaScript

### Dataset

* PlantVillage

---

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/PlantDoc.git
cd PlantDoc
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
cd plantdoc-chatbot/backend
pip install -r requirements.txt
```

---

##  Model Files

The trained model artifacts are **not included in this repository** because of their size.

### Text Model

The text model is expected at:

```text
plantdoc-chatbot/backend/text_model/
```

### Image Model

The image inference code expects:

```text
plantdoc-chatbot/models/
```

containing:

```text
plantdoc_image_model.pt
class_names.json
```

These model artifacts need to be available locally before running the complete inference application.

---

##  Running the Backend

Navigate to the backend directory:

```bash
cd plantdoc-chatbot/backend
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The backend provides the image prediction, text prediction, and combined chat endpoints.

---

##  Repository Exclusions

The following files and directories are intentionally excluded from GitHub:

```text
venv/
plantdoc-chatbot/backend/venv/
dataset/
plantdoc-chatbot/backend/text_model/
__pycache__/
*.py[cod]
.DS_Store
```

This keeps the repository focused on the source code while excluding:

* Local Python environments
* Large datasets
* Downloaded model files
* Python cache files
* macOS system files

---

## Future Enhancements

* Improved image classification model
* Alignment of training and inference pipelines
* Support for additional crops and diseases
* Cloud deployment
* Mobile application integration
* Improved conversational assistance
* Plant health recommendations
* Disease history and analytics
* Multi-language support

---

##Proj ect Objective

The objective of PlantDoc AI is to demonstrate how **computer vision, deep learning, and natural language processing** can be combined to create an AI-assisted plant disease detection and assistance system.

The project brings together:

* Machine learning model development
* Image classification
* Natural language processing
* Model inference
* REST API development
* Frontend integration

---

## Author

**Pavitraa V**

BE — Artificial Intelligence & Machine Learning

---

## License

This project is intended for educational and research purposes.
