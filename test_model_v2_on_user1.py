import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_v2 import SimpleCNNv2


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    test_dir = "data/user1_animals/test"
    dataset = datasets.ImageFolder(test_dir, transform=tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    num_classes = len(dataset.classes)  # should be 5: cat, dog, elephant, horse, lion
    print("Classes:", dataset.classes)

    model = SimpleCNNv2(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/model_v2.pth", map_location=device))

    acc = evaluate(model, loader, device)

    os.makedirs("results", exist_ok=True)
    out_path = "results/test_v2_user1.json"
    with open(out_path, "w") as f:
        json.dump({"test_acc": acc}, f, indent=2)

    print("Test accuracy (model_v2 on user1 animals):", acc)
    print("Saved to", out_path)
