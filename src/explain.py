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
