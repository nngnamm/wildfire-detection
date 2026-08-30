"""
main.py
-------
Orchestration script for the wildfire detection pipeline. Ties together
data loading, model construction, training, evaluation, and Grad-CAM
explainability.

Usage:
    python main.py --mode train    --data_dir dataset --epochs 10
    python main.py --mode evaluate --data_dir dataset
    python main.py --mode explain  --data_dir dataset --image_path path/to/image.jpg
    python main.py --mode grid     --data_dir dataset --n_per_class 4
    python main.py --mode failures --data_dir dataset --n_per_type 4
    python main.py --mode all      --data_dir dataset --epochs 10
"""

import argparse
import os

import torch
from PIL import Image

from src.dataset import get_dataloaders, get_transforms
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.explain import explain_prediction, generate_comparison_grid, find_and_explain_misclassified


def parse_args():
    parser = argparse.ArgumentParser(description="Wildfire Detection Pipeline")
    parser.add_argument("--mode", type=str, default="all",
                         choices=["train", "evaluate", "explain", "grid", "failures", "all"],
                         help="Which stage of the pipeline to run.")
    parser.add_argument("--data_dir", type=str, default="dataset",
                         help="Path to the dataset root (train/valid/test subfolders).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--image_path", type=str, default=None,
                         help="Path to a single image for Grad-CAM explanation. "
                              "If omitted, the first test-set image is used.")
    parser.add_argument("--n_per_class", type=int, default=4,
                         help="Number of images per class to include in the "
                              "--mode grid comparison (default: 4).")
    parser.add_argument("--n_per_type", type=int, default=4,
                         help="Number of false negatives / false positives to "
                              "include in the --mode failures comparison (default: 4).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    model = build_model(freeze_backbone=True, device=device)

    if args.mode in ("train", "all"):
        model, history = train_model(
            model, train_loader, val_loader, device=device,
            num_epochs=args.epochs, learning_rate=args.lr,
            checkpoint_path=args.checkpoint,
        )

    if args.mode in ("evaluate", "explain", "grid", "failures") and os.path.exists(args.checkpoint):
        # Load the best saved weights if we didn't just train in this run.
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[main] Loaded weights from '{args.checkpoint}'")

    if args.mode in ("evaluate", "all"):
        evaluate_model(model, test_loader, device=device)

    if args.mode in ("explain", "all"):
        _, eval_transform = get_transforms()

        if args.image_path is not None:
            image_path = args.image_path
        else:
            # Grab the path of the first image in the test dataset for a
            # convenient, no-argument demo.
            image_path, _ = test_loader.dataset.samples[0]

        image = Image.open(image_path).convert("RGB")
        input_tensor = eval_transform(image).unsqueeze(0).to(device)
        explain_prediction(model, input_tensor, save_path="gradcam_output.png")

    if args.mode == "grid":
        _, eval_transform = get_transforms()
        generate_comparison_grid(
            model, test_loader.dataset, eval_transform, class_to_idx,
            device=device, n_per_class=args.n_per_class,
            save_path="gradcam_grid.png",
        )

    if args.mode == "failures":
        _, eval_transform = get_transforms()
        find_and_explain_misclassified(
            model, test_loader.dataset, eval_transform, class_to_idx,
            device=device, n_per_type=args.n_per_type,
            save_path="gradcam_failures.png",
        )


if __name__ == "__main__":
    main()
