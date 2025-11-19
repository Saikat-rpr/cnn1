import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_v1 import SimpleCNN   # from User1


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms same as v2 eval
    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    test_dir = "data/user2_plants/test"
    ds = datasets.ImageFolder(test_dir, tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    num_classes = 5  # plant classes
    model = SimpleCNN(num_classes=num_classes).to(device)

    model.load_state_dict(torch.load("models/model_v1.pth", map_location=device))

    acc = evaluate(model, loader, device)

    os.makedirs("results", exist_ok=True)
    with open("results/test_v1_user2.json", "w") as f:
        json.dump({"test_acc": acc}, f, indent=2)

    print("Test accuracy (model_v1 on user2 plants):", acc)
