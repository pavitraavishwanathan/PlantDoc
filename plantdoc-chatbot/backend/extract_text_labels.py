from transformers import AutoConfig
import json

config = AutoConfig.from_pretrained(".")

# id2label is stored in trained models
labels = [config.id2label[i] for i in range(len(config.id2label))]

with open("text_class_names.json", "w") as f:
    json.dump(labels, f, indent=2)

print("✅ Extracted", len(labels), "text classes")
print(labels)
