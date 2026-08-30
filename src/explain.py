"""
explain.py
----------
Grad-CAM implementation for model explainability. Hooks into the last
convolutional block of ResNet50 (layer4) and produces a heatmap
showing which regions of the input image most influenced the model's
wildfire prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for a binary classifier with a single sigmoid output.

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.layer4[-1])
        heatmap, prob = cam.generate(input_tensor)
        cam.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture the forward activations and the
        # gradients flowing back into them during backpropagation.
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor):
        """
        Runs a forward + backward pass on a single image tensor
        (shape: 1 x 3 x H x W) and returns a normalized [0, 1] heatmap
        of shape (H, W) plus the model's predicted probability.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)          # shape (1, 1), sigmoid probability
        prob = output.item()

        self.model.zero_grad()
        # Backpropagate the scalar output directly: since the model's
        # own output already IS the class probability (single logit +
        # sigmoid), we use it as the target score for Grad-CAM.
        output.backward(retain_graph=True)

        gradients = self.gradients        # shape (1, C, h, w)
        activations = self.activations    # shape (1, C, h, w)

        # Global-average-pool the gradients over spatial dimensions to
        # get one importance weight per channel (the standard Grad-CAM
        # weighting scheme).
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # shape (1, C, 1, 1)

        # Weighted combination of activation maps, followed by ReLU
        # (Grad-CAM only cares about features that positively influence
        # the target class).
        cam = (weights * activations).sum(dim=1, keepdim=True)  # shape (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to the input image's spatial size.
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1] for visualization.
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, prob


def overlay_heatmap_on_image(
    heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto the original RGB image.

    Args:
        heatmap: 2D array in [0, 1], same H, W as the image.
        original_image: HxWx3 RGB image, values in [0, 255], dtype uint8.
        alpha: blending factor for the heatmap.

    Returns:
        HxWx3 RGB uint8 image with the heatmap overlaid.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = (colored_heatmap.astype(np.float32) * alpha +
               original_image.astype(np.float32) * (1 - alpha))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized (ImageNet mean/std) CHW tensor back into an
    HxWx3 uint8 RGB numpy image suitable for visualization/overlay.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    image = (image * std) + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def explain_prediction(model, input_tensor: torch.Tensor, save_path: str = "gradcam_output.png"):
    """
    High-level convenience function: generates a Grad-CAM heatmap for a
    single input image tensor (already normalized, shape (1, 3, H, W)),
    overlays it, and saves the result to `save_path`.
    """
    target_layer = model.backbone.layer4[-1]
    cam = GradCAM(model, target_layer)

    heatmap, prob = cam.generate(input_tensor)
    cam.remove_hooks()

    original_image = denormalize_image(input_tensor.squeeze(0))
    overlay = overlay_heatmap_on_image(heatmap, original_image)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay_bgr)

    label = "WILDFIRE" if prob >= 0.5 else "NO WILDFIRE"
    print(f"[explain] Predicted probability of wildfire: {prob:.4f} -> {label}")
    print(f"[explain] Grad-CAM overlay saved to '{save_path}'")

    return heatmap, prob


def generate_comparison_grid(
    model,
    test_dataset,
    eval_transform,
    class_to_idx: dict,
    device: str,
    n_per_class: int = 4,
    save_path: str = "gradcam_grid.png",
):
    """
    Runs Grad-CAM on several wildfire images AND several no-wildfire images
    from the test set, side by side in one grid image: top row = original
    image, bottom row = Grad-CAM overlay, with the true label and the
    model's predicted probability captioned on each.

    This exists because a single Grad-CAM image tells you very little —
    it could be a lucky example either way. Looking at several images from
    both classes at once lets you check whether the model's attention is
    *consistent* (e.g. always on terrain/vegetation patterns for wildfire
    images) rather than judging from one anecdotal heatmap.

    Args:
        model: trained WildfireResNet50 model.
        test_dataset: the underlying torchvision ImageFolder dataset
                       (e.g. test_loader.dataset) — used to find file
                       paths per class without loading everything into
                       memory first.
        eval_transform: the same resize/normalize transform used for
                         evaluation (from get_transforms()).
        class_to_idx: dict like {'nowildfire': 0, 'wildfire': 1}, used to
                       label images correctly regardless of which index
                       ImageFolder assigned to which class name.
        device: 'cuda' or 'cpu'.
        n_per_class: how many images to sample from each class.
        save_path: where to save the resulting grid image.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    wildfire_idx = class_to_idx["wildfire"]
    nowildfire_idx = class_to_idx["nowildfire"]

    wildfire_samples = [s for s in test_dataset.samples if s[1] == wildfire_idx][:n_per_class]
    nowildfire_samples = [s for s in test_dataset.samples if s[1] == nowildfire_idx][:n_per_class]
    all_samples = wildfire_samples + nowildfire_samples

    if len(all_samples) == 0:
        raise ValueError("No samples found for the given classes — check class_to_idx and test_dataset.")

    n = len(all_samples)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))
    # If there's only one column, axes won't be 2D — normalize its shape
    # so the indexing below (axes[row, col]) always works.
    if n == 1:
        axes = axes.reshape(2, 1)

    target_layer = model.backbone.layer4[-1]

    for i, (path, true_idx) in enumerate(all_samples):
        image = Image.open(path).convert("RGB")
        input_tensor = eval_transform(image).unsqueeze(0).to(device)

        cam = GradCAM(model, target_layer)
        heatmap, prob = cam.generate(input_tensor)
        cam.remove_hooks()

        original_image = denormalize_image(input_tensor.squeeze(0))
        overlay = overlay_heatmap_on_image(heatmap, original_image)

        true_label = idx_to_class[true_idx]
        pred_label = "wildfire" if prob >= 0.5 else "nowildfire"
        correct = "\u2713" if pred_label == true_label else "\u2717"

        axes[0, i].imshow(original_image)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"True: {true_label}", fontsize=9)

        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Pred: {pred_label} ({prob:.2f}) {correct}", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[explain] Comparison grid ({len(wildfire_samples)} wildfire + "
          f"{len(nowildfire_samples)} nowildfire) saved to '{save_path}'")


@torch.no_grad()
def _predict_all(model, test_dataset, eval_transform, device, batch_size: int = 32):
    """
    Runs the model over every image in test_dataset and returns a list of
    (file_path, true_label_idx, predicted_probability) for each one. Used
    by find_and_explain_misclassified() to locate real failure cases
    without needing gradients (no backward pass here — this is just a
    fast scan to find which files the model got wrong).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    results = []
    sample_idx = 0

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        probs = model(images).cpu().numpy().flatten()

        for prob, label in zip(probs, labels.numpy()):
            path, _ = test_dataset.samples[sample_idx]
            results.append((path, int(label), float(prob)))
            sample_idx += 1

    return results


def find_and_explain_misclassified(
    model,
    test_dataset,
    eval_transform,
    class_to_idx: dict,
    device: str,
    n_per_type: int = 4,
    save_path: str = "gradcam_failures.png",
):
    """
    Finds real false negatives (actual wildfire, predicted no-wildfire)
    and false positives (actual no-wildfire, predicted wildfire) in the
    test set, runs Grad-CAM on a sample of each, and saves them in one
    comparison grid.

    Unlike testing on an image found online, every image here comes from
    the same distribution the model was trained and evaluated on, so any
    pattern found is a genuine model limitation rather than an artifact
    of a different image source, region, or color processing.

    Args:
        model: trained WildfireResNet50 model.
        test_dataset: the underlying torchvision ImageFolder dataset
                       (e.g. test_loader.dataset).
        eval_transform: the same resize/normalize transform used for
                         evaluation (from get_transforms()).
        class_to_idx: dict like {'nowildfire': 0, 'wildfire': 1}.
        device: 'cuda' or 'cpu'.
        n_per_type: how many false negatives and false positives to show
                    (skipped if fewer exist than requested).
        save_path: where to save the resulting grid image.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    wildfire_idx = class_to_idx["wildfire"]
    nowildfire_idx = class_to_idx["nowildfire"]

    print("[explain] Scanning test set for misclassified images (this runs the "
          "full test set through the model once, no gradients — should be much "
          "faster than training)...")
    all_results = _predict_all(model, test_dataset, eval_transform, device)

    false_negatives = [(p, prob) for p, label, prob in all_results
                        if label == wildfire_idx and prob < 0.5][:n_per_type]
    false_positives = [(p, prob) for p, label, prob in all_results
                        if label == nowildfire_idx and prob >= 0.5][:n_per_type]

    print(f"[explain] Found {len(false_negatives)} false negative(s) and "
          f"{len(false_positives)} false positive(s) to display (showing up to "
          f"{n_per_type} of each).")

    all_samples = (
        [(p, wildfire_idx, prob) for p, prob in false_negatives] +
        [(p, nowildfire_idx, prob) for p, prob in false_positives]
    )

    if len(all_samples) == 0:
        print("[explain] No misclassified images found — nothing to plot.")
        return

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    n = len(all_samples)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    target_layer = model.backbone.layer4[-1]

    for i, (path, true_idx, precomputed_prob) in enumerate(all_samples):
        image = Image.open(path).convert("RGB")
        input_tensor = eval_transform(image).unsqueeze(0).to(device)

        cam = GradCAM(model, target_layer)
        heatmap, prob = cam.generate(input_tensor)  # re-run with gradients enabled for Grad-CAM
        cam.remove_hooks()

        original_image = denormalize_image(input_tensor.squeeze(0))
        overlay = overlay_heatmap_on_image(heatmap, original_image)

        true_label = idx_to_class[true_idx]
        error_type = "False Negative" if true_idx == wildfire_idx else "False Positive"

        axes[0, i].imshow(original_image)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"True: {true_label}\n({error_type})", fontsize=9)

        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Pred prob: {prob:.2f}", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[explain] Failure-case grid saved to '{save_path}'")
