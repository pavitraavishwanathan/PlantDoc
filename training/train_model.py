import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ===== PATHS =====
DATA_DIR = "../dataset/PlantVillage"
MODEL_PATH = "../backend/model.pth"

# ===== CONFIG =====
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = "cpu"

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== DATASET =====
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(dataset.classes)
print("Classes:", dataset.classes)

# ===== MODEL =====
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

# ===== TRAINING =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    running_loss = 0.0
    print(f"\n🔁 Starting Epoch {epoch+1}/{EPOCHS}")

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} completed - Total Loss: {running_loss:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "classes": dataset.classes
}, MODEL_PATH)

print("✅ model.pth saved in backend/")
