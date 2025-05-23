# ultimate_morph_generator/dgo_oracle/explainability_tools.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Callable, Dict
import cv2  # For heatmap visualization

from ..config import get_config
from ..utilities.image_utils import preprocess_image_for_dgo, tensor_to_np_cv
from ..utilities.type_definitions import CvImage, PILImage, TorchImageTensor
from .dgo_model_handler import DGOModelHandler
from ..utilities.logging_config import setup_logging

logger = setup_logging()


# --- Grad-CAM specific imports (example, might use a library like 'captum') ---
# from captum.attr import LayerGradCam, GuidedBackprop # Example if using Captum

class DGOExplainer:
    """
    Provides tools to explain DGO model's decisions.
    Wraps techniques like vanilla gradients, Grad-CAM, etc.
    """

    def __init__(self, model_handler: DGOModelHandler):
        self.model_handler = model_handler
        self.model = model_handler.model
        self.device = model_handler.device
        self.data_cfg = model_handler.data_cfg  # For image preprocessing info

    def get_vanilla_saliency(self, image_input: Union[CvImage, PILImage, str, torch.Tensor],
                             target_class_idx: Optional[int] = None) -> np.ndarray:
        """
        Computes vanilla saliency map (absolute value of gradients of the target class score w.r.t. input).
        If target_class_idx is None, uses the predicted class.
        Returns a 2D numpy array (H, W) representing the saliency map.
        """
        if not isinstance(image_input, torch.Tensor):
            input_tensor = preprocess_image_for_dgo(
                image_input,
                target_size=self.data_cfg.target_image_size,
                grayscale=self.data_cfg.grayscale_input,
                device=self.device
            )
        else:
            input_tensor = image_input.to(self.device)
            if input_tensor.ndim == 3: input_tensor = input_tensor.unsqueeze(0)

        input_tensor.requires_grad = True
        self.model.zero_grad()

        original_training_state = self.model.training
        self.model.eval()  # Ensure eval mode

        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        score_for_target_class = logits[0, target_class_idx]
        score_for_target_class.backward()

        saliency = input_tensor.grad.data.abs().squeeze()  # (C, H, W) or (H, W)

        if saliency.ndim == 3:  # Multi-channel (e.g., color image)
            saliency = saliency.max(dim=0)[0]  # Max across channels, or mean

        saliency_np = saliency.cpu().numpy()

        self.model.train(original_training_state)  # Restore mode
        return saliency_np

    def get_grad_cam(self,
                     image_input: Union[CvImage, PILImage, str, torch.Tensor],
                     target_layer: Union[nn.Module, str],  # Actual layer module or its name
                     target_class_idx: Optional[int] = None
                     ) -> Optional[np.ndarray]:
        """
        Computes Grad-CAM heatmap.
        This is a simplified placeholder. A full implementation would use hooks
        to get gradients and activations from the target_layer.
        Libraries like 'captum' or 'grad-cam' are recommended for robust Grad-CAM.

        Args:
            image_input: The input image.
            target_layer: The convolutional layer to use for Grad-CAM. Can be the layer itself
                          or its name (if model supports named module access).
            target_class_idx: The class index for which to generate the CAM. If None, uses predicted class.

        Returns:
            A 2D numpy array (H, W) of the heatmap, or None if failed.
        """
        logger.warning(
            "Grad-CAM is a placeholder here. For full functionality, integrate a library like 'captum' or 'pytorch-grad-cam'.")

        # --- Simplified conceptual steps (not a full working Grad-CAM) ---
        if isinstance(target_layer, str):
            try:
                # Find the layer by name
                module_found = None
                for name, mod in self.model.named_modules():
                    if name == target_layer:
                        module_found = mod
                        break
                if module_found is None:
                    logger.error(f"Grad-CAM: Target layer '{target_layer}' not found by name.")
                    return None
                target_layer_module = module_found
            except Exception as e:
                logger.error(f"Error accessing target layer '{target_layer}': {e}")
                return None
        elif isinstance(target_layer, nn.Module):
            target_layer_module = target_layer
        else:
            logger.error("Grad-CAM: target_layer must be an nn.Module or its name string.")
            return None

        # 1. Preprocess input
        if not isinstance(image_input, torch.Tensor):
            input_tensor = preprocess_image_for_dgo(image_input, self.data_cfg.target_image_size,
                                                    self.data_cfg.grayscale_input, self.device)
        else:
            input_tensor = image_input.to(self.device)
            if input_tensor.ndim == 3: input_tensor = input_tensor.unsqueeze(0)
        input_tensor.requires_grad_(True)

        self.model.eval()
        self.model.zero_grad()

        # 2. Hook to get activations and gradients from target_layer
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        forward_handle = target_layer_module.register_forward_hook(forward_hook)
        backward_handle = target_layer_module.register_backward_hook(backward_hook)

        # 3. Forward pass
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        # 4. Backward pass for the target class
        score = logits[:, target_class_idx].sum()  # Sum if batch > 1, or just take [0]
        score.backward()

        # Clean up hooks
        forward_handle.remove()
        backward_handle.remove()

        if not activations or not gradients:
            logger.error("Grad-CAM: Failed to capture activations or gradients.")
            return None

        # 5. Compute Grad-CAM
        # activations[0]: (B, C_layer, H_layer, W_layer)
        # gradients[0]: (B, C_layer, H_layer, W_layer)
        act_val = activations[0].squeeze()  # Assuming B=1, (C_layer, H_l, W_l)
        grad_val = gradients[0].squeeze()  # (C_layer, H_l, W_l)

        # Global average pooling of gradients (alpha_k weights)
        weights = torch.mean(grad_val, dim=(1, 2), keepdim=True)  # (C_layer, 1, 1)

        # Weighted sum of activation maps
        cam = torch.sum(weights * act_val, dim=0)  # (H_l, W_l)
        cam = F.relu(cam)  # Apply ReLU

        # Normalize and resize to original image size
        cam_np = cam.cpu().numpy()
        cam_np -= np.min(cam_np)
        if np.max(cam_np) > 0:  # Avoid division by zero
            cam_np /= np.max(cam_np)

        # Resize to original image dimensions (H, W from data_cfg)
        # input_tensor shape is (B, C, H, W)
        original_h, original_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam_np, (original_w, original_h))

        return heatmap

    # Interface for other methods like SHAP or LIME (would require external libraries)
    # def get_shap_values(...) -> ...
    # def get_lime_explanation(...) -> ...

    @staticmethod
    def overlay_heatmap_on_image(image_np: np.ndarray, heatmap_np: np.ndarray,
                                 alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Overlays a heatmap onto an image."""
        if image_np.ndim == 2:  # Grayscale image
            image_np_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            image_np_color = image_np.copy()

        if image_np_color.dtype != np.uint8:  # Ensure uint8 for colormap application
            image_np_color = np.clip(image_np_color, 0, 255).astype(np.uint8)

        heatmap_norm = (heatmap_np * 255).astype(np.uint8)  # Normalize to 0-255
        heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)

        # Ensure heatmap_color has same size as image_np_color
        if heatmap_color.shape[:2] != image_np_color.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (image_np_color.shape[1], image_np_color.shape[0]))

        overlaid_image = cv2.addWeighted(image_np_color, 1 - alpha, heatmap_color, alpha, 0)
        return overlaid_image


if __name__ == "__main__":
    # --- Test DGOExplainer ---
    from ..config import SystemConfig

    temp_sys_cfg_data = {  # Using BaseCNN for testing
        "project_name": "ExplainerTest",
        "data_management": {"target_image_size": (28, 28), "grayscale_input": True},
        "dgo_oracle": {
            "model_architecture": "BaseCNN", "num_classes": 10,
            "pretrained_model_path": None,
            "feature_extraction_layer_name": "relu3",
            "base_cnn_fc_features": 64
        },
        "logging": {"level": "DEBUG"}
    }
    from ..config import _config_instance

    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data)
    cfg = get_config()
    logger = setup_logging()
    device = cfg.get_actual_device()

    dgo_handler = DGOModelHandler(cfg.dgo_oracle, cfg.data_management, device)
    explainer = DGOExplainer(dgo_handler)

    dummy_cv_image = np.random.randint(0, 256, cfg.data_management.target_image_size, dtype=np.uint8)
    cv2.putText(dummy_cv_image, "E", (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200), 2)

    # Test Vanilla Saliency
    print("\n--- Testing Vanilla Saliency ---")
    saliency_map = explainer.get_vanilla_saliency(dummy_cv_image, target_class_idx=3)
    print(f"Saliency map shape: {saliency_map.shape}")
    assert saliency_map.shape == cfg.data_management.target_image_size

    # Visualize Saliency (optional, requires matplotlib or display window)
    # import matplotlib.pyplot as plt
    # plt.imshow(saliency_map, cmap='hot'); plt.title("Vanilla Saliency"); plt.show()
    saliency_overlay = explainer.overlay_heatmap_on_image(dummy_cv_image, saliency_map, alpha=0.6)
    # cv2.imshow("Saliency Overlay", saliency_overlay); cv2.waitKey(0); cv2.destroyAllWindows()
    print("Vanilla Saliency test completed (visual check recommended).")

    # Test Grad-CAM (placeholder functionality)
    print("\n--- Testing Grad-CAM (Placeholder) ---")
    # For BaseCNN, 'conv2' or 'relu2' are suitable target layers before pooling
    # Or even 'pool2' output (activations before flatten)
    # The target_layer name must match a name in model.named_modules()
    # Let's use 'relu2' as it's a common conv layer output in BaseCNN.
    target_conv_layer_name_basecnn = "relu2"
    # For ResNet, it might be 'layer4' or a specific block's conv.

    # Verify the layer name exists
    found = False
    for name, _ in dgo_handler.model.named_modules():
        if name == target_conv_layer_name_basecnn:
            found = True;
            break
    if not found:
        print(
            f"Could not find layer '{target_conv_layer_name_basecnn}' for Grad-CAM test with BaseCNN. Skipping Grad-CAM test.")
    else:
        grad_cam_heatmap = explainer.get_grad_cam(dummy_cv_image,
                                                  target_layer=target_conv_layer_name_basecnn,
                                                  target_class_idx=3)
        if grad_cam_heatmap is not None:
            print(f"Grad-CAM heatmap shape: {grad_cam_heatmap.shape}")
            assert grad_cam_heatmap.shape == cfg.data_management.target_image_size
            grad_cam_overlay = explainer.overlay_heatmap_on_image(dummy_cv_image, grad_cam_heatmap)
            # cv2.imshow("Grad-CAM Overlay", grad_cam_overlay); cv2.waitKey(0); cv2.destroyAllWindows()
            print("Grad-CAM test completed (visual check recommended).")
        else:
            print("Grad-CAM heatmap generation failed (as expected for placeholder or if layer not found).")

    print("\nDGOExplainer tests completed.")