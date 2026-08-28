"""
evaluate.py
-----------
Computes test-set evaluation metrics: Accuracy, Precision, Recall,
F1-Score, and Confusion Matrix. For a wildfire detection system, False
Negatives (missed fires) are the most dangerous error type, so recall
is highlighted explicitly in the printed report.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@torch.no_grad()
def evaluate_model(model, test_loader, device: str, threshold: float = 0.5):
    """
    Runs inference over the entire test set and computes classification
    metrics.

    Args:
        model: trained WildfireResNet50 model.
        test_loader: DataLoader for the test split.
        device: 'cuda' or 'cpu'.
        threshold: probability threshold used to convert sigmoid output
                   into a binary class prediction.

    Returns:
        metrics: dict containing accuracy, precision, recall, f1, and
                 the raw confusion matrix (as a numpy array).
    """
    model.eval()

    all_labels = []
    all_preds = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).cpu().numpy().flatten()
        preds = (outputs >= threshold).astype(int)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

    _print_report(metrics, cm)
    return metrics


def _print_report(metrics: dict, cm: np.ndarray):
    print("\n===== Test Set Evaluation =====")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}  <-- critical: minimizes missed wildfires (False Negatives)")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix (rows = actual, cols = predicted):")
    print("                 Pred: No-Fire   Pred: Fire")
    if cm.shape == (2, 2):
        print(f"Actual: No-Fire      {cm[0, 0]:>6d}         {cm[0, 1]:>6d}")
        print(f"Actual: Fire         {cm[1, 0]:>6d}         {cm[1, 1]:>6d}")
        fn = cm[1, 0]
        if fn > 0:
            print(f"\n[!] Warning: {fn} False Negative(s) — actual wildfires classified as 'no fire'.")
    else:
        print(cm)
    print("================================\n")
