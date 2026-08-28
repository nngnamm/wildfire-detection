"""
train.py
--------
Training and validation loop for the wildfire detection model.
Uses Binary Cross-Entropy loss (nn.BCELoss) and the Adam optimizer,
optimizing only the unfrozen classification head.
"""

import copy
import time

import torch
import torch.nn as nn

from src.model import WildfireResNet50


def train_model(
    model: WildfireResNet50,
    train_loader,
    val_loader,
    device: str,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "best_model.pth",
):
    """
    Trains `model` for `num_epochs`, evaluating on `val_loader` after
    every epoch and keeping the weights with the best validation
    accuracy (saved to `checkpoint_path`).

    Returns:
        model: the model loaded with the best validation weights.
        history: dict with per-epoch train/val loss and accuracy.
    """
    criterion = nn.BCELoss()
    # Only the classification head has requires_grad=True since the
    # backbone was frozen in model.py, so this only updates the head.
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ---------------- Training phase ----------------
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # shape (B, 1) to match model output

            optimizer.zero_grad()
            outputs = model(images)          # probabilities in [0, 1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (outputs >= 0.5).float()
            running_correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # ---------------- Validation phase ----------------
        val_loss, val_acc = _evaluate_loss_and_accuracy(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, checkpoint_path)
            print(f"  -> New best model saved (val_acc={val_acc:.4f}) to '{checkpoint_path}'")

    model.load_state_dict(best_model_weights)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    return model, history


def _evaluate_loss_and_accuracy(model, data_loader, criterion, device):
    """Helper used during training to compute validation loss/accuracy."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (outputs >= 0.5).float()
            running_correct += (preds == labels).sum().item()
            total += images.size(0)

    return running_loss / total, running_correct / total
