import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_v2 import SimpleCNNv2


def build_dataloaders(root, batch_size):
    """
    Data loaders for User 2 (PlantVillage subset).
    Structure:
      root/train/<class>/
      root/val/<class>/
      root/test/<class>/
    """
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(root, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(root, "test"),  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


def loop_one_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_training(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_root, args.batch_size
    )
    num_classes = len(class_names)
    print("Classes:", class_names)

    model = SimpleCNNv2(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {
        "config": {
            "model_name": "model_v2",
            "dataset": "plantvillage_user2",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "data_root": args.data_root,
        },
        "class_names": class_names,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = loop_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = loop_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val  : loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_out)
            print(f"  >> Saved new best model to {args.model_out}")

    # evaluate best model on test set
    model.load_state_dict(torch.load(args.model_out, map_location=device))
    test_loss, test_acc = loop_one_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )

    total_time = time.time() - start

    history["best_val_acc"] = best_val_acc
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["training_time_sec"] = total_time

    with open(args.metrics_out, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTest loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    print(f"Metrics saved to {args.metrics_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="User 2: train model_v2 on PlantVillage subset")
    parser.add_argument("--data-root", type=str, default="data/user2_plants")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-out", type=str, default="models/model_v2.pth")
    parser.add_argument("--metrics-out", type=str, default="results/metrics_v2.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
