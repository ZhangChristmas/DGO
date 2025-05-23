# ultimate_morph_generator/utilities/image_utils.py
import cv2
import numpy as np
import torch
from torchvision import transforms  # type: ignore
from PIL import Image  # type: ignore
from typing import Tuple, Union, Optional

from .type_definitions import CvImage, TorchImageTensor, PILImage
from ..config import get_config  # 使用相对导入


# --- Conversion Functions ---

def cv_to_pil(cv_image: CvImage) -> PILImage:
    """Converts an OpenCV image (NumPy array) to a PIL Image."""
    if cv_image.ndim == 2:  # Grayscale
        return Image.fromarray(cv_image)
    elif cv_image.ndim == 3 and cv_image.shape[2] == 3:  # Color (BGR)
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    elif cv_image.ndim == 3 and cv_image.shape[2] == 1:  # Grayscale with channel dim
        return Image.fromarray(cv_image.squeeze(axis=2))
    else:
        raise ValueError(f"Unsupported OpenCV image format: shape {cv_image.shape}")


def pil_to_cv(pil_image: PILImage) -> CvImage:
    """Converts a PIL Image to an OpenCV image (NumPy array)."""
    img_np = np.array(pil_image)
    if pil_image.mode == 'RGB':
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif pil_image.mode == 'L':  # Grayscale
        return img_np
    elif pil_image.mode == 'RGBA':  # Handle alpha if needed, here we convert to BGR
        return cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:  # Potentially other modes like P, CMYK etc.
        # Convert to RGB first then to BGR for simplicity
        rgb_pil = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)


def np_to_tensor(image_np: np.ndarray, device: Optional[torch.device] = None) -> TorchImageTensor:
    """
    Converts a NumPy image (H, W, C) or (H, W) to a PyTorch tensor (1, C, H, W).
    Assumes pixel values are in [0, 255] uint8 or [0, 1] float.
    Normalizes to [0, 1] float32.
    """
    cfg = get_config()
    target_device = device if device else cfg.get_actual_device()

    if image_np.dtype == np.uint8:
        image_np = image_np.astype(np.float32) / 255.0

    if image_np.ndim == 2:  # Grayscale (H, W) -> (1, 1, H, W)
        image_np = np.expand_dims(image_np, axis=0)  # (1, H, W)
        image_np = np.expand_dims(image_np, axis=0)  # (1, 1, H, W)
    elif image_np.ndim == 3:  # Color (H, W, C) or Grayscale (H, W, 1)
        if image_np.shape[2] == 1:  # (H,W,1) -> (1,1,H,W)
            image_np = image_np.transpose((2, 0, 1))  # (1, H, W)
            image_np = np.expand_dims(image_np, axis=0)  # (1, 1, H, W)
        elif image_np.shape[2] == 3:  # (H,W,C) -> (1,C,H,W)
            # Assuming input is (H, W, C), need to convert to (C, H, W)
            image_np = image_np.transpose((2, 0, 1))  # (C, H, W)
            image_np = np.expand_dims(image_np, axis=0)  # (1, C, H, W)
        else:
            raise ValueError(f"Unsupported NumPy image shape for tensor conversion: {image_np.shape}")
    else:
        raise ValueError(f"Unsupported NumPy image ndim for tensor conversion: {image_np.ndim}")

    return torch.from_numpy(image_np).float().to(target_device)


def tensor_to_np_cv(tensor: TorchImageTensor, denormalize: bool = False) -> CvImage:
    """
    Converts a PyTorch tensor (B, C, H, W) or (C, H, W) to a NumPy OpenCV image (H, W, C) or (H, W).
    If batch size B > 1, returns the first image in the batch.
    Assumes tensor values are in [0, 1] or [-1, 1] if normalized.
    Outputs uint8 image [0, 255].
    """
    if tensor.ndim == 4 and tensor.shape[0] > 1:  # Batch of images
        tensor = tensor[0]  # Take the first image
    elif tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # (C, H, W)

    # tensor is now (C, H, W)
    img_np = tensor.cpu().detach().numpy()

    if denormalize:  # Example denormalization if mean/std were used.
        # This needs to match the normalization used. For simple [0,1] -> [0,255]:
        pass  # Already handled by * 255 if input is [0,1]
        # If it was normalized with mean/std, e.g., for ImageNet:
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # for i in range(img_np.shape[0]): # For each channel
        #     img_np[i] = img_np[i] * std[i] + mean[i]

    if img_np.shape[0] == 1:  # Grayscale (1, H, W) -> (H, W)
        img_np = img_np.squeeze(0)
    elif img_np.shape[0] == 3:  # Color (3, H, W) -> (H, W, 3) for OpenCV
        img_np = img_np.transpose((1, 2, 0))
    else:
        raise ValueError(f"Unsupported tensor shape for NumPy conversion: {tensor.shape}")

    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    # If it was RGB and needs to be BGR for OpenCV display/saving with cv2
    # if img_np.ndim == 3 and img_np.shape[2] == 3:
    #    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Assuming tensor was RGB

    return img_np


# --- Preprocessing for DGO ---
def preprocess_image_for_dgo(image_input: Union[CvImage, PILImage, str],
                             target_size: Tuple[int, int],
                             grayscale: bool,
                             device: Optional[torch.device] = None) -> TorchImageTensor:
    """
    Comprehensive preprocessing for DGO input.
    Handles various input types (path, PIL, CvImage), resizes, converts to grayscale if needed,
    normalizes, and converts to a PyTorch tensor.
    """
    cfg = get_config()
    dgo_cfg = cfg.dgo_oracle

    if isinstance(image_input, str):  # Path to image
        pil_image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):  # CvImage
        pil_image = cv_to_pil(image_input)
    elif isinstance(image_input, Image.Image):  # PILImage
        pil_image = image_input
    else:
        raise TypeError(f"Unsupported input type for DGO preprocessing: {type(image_input)}")

    # Standard transformations
    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC))  # High quality resize
    transform_list.append(transforms.ToTensor())  # Converts PIL [0,255] to Tensor [0,1]

    # MNIST-like normalization (example, DGO might be trained with different normalization)
    # This should be configurable or based on DGO's training regime.
    # For now, let's assume a simple [0,1] range from ToTensor is fine, or a generic normalization.
    if grayscale:
        # Example for MNIST-like data if DGO was trained on it
        # transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        pass  # Often, for generated data, keeping [0,1] is simpler unless DGO is very specific
    else:  # Color
        # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        pass

    composed_transforms = transforms.Compose(transform_list)
    processed_tensor = composed_transforms(pil_image)

    # Add batch dimension and send to device
    target_device = device if device else cfg.get_actual_device()
    return processed_tensor.unsqueeze(0).to(target_device)


# --- Image Augmentation/Display ---
def pad_image_to_square(cv_image: CvImage, fill_color: Union[int, Tuple[int, int, int]] = 0) -> CvImage:
    """Pads an image to make it square, placing the original image in the center."""
    h, w = cv_image.shape[:2]
    if h == w:
        return cv_image

    target_dim = max(h, w)

    if cv_image.ndim == 2:  # Grayscale
        padded_image = np.full((target_dim, target_dim), fill_color, dtype=cv_image.dtype)
    else:  # Color
        padded_image = np.full((target_dim, target_dim, cv_image.shape[2]), fill_color, dtype=cv_image.dtype)

    pad_top = (target_dim - h) // 2
    pad_left = (target_dim - w) // 2

    padded_image[pad_top:pad_top + h, pad_left:pad_left + w] = cv_image
    return padded_image


def standardize_image(image: CvImage, target_size: Tuple[int, int],
                      grayscale: bool = True,
                      invert_if_dark_on_light: bool = False,
                      threshold_for_inversion_check: int = 128) -> CvImage:
    """
    Standardizes an image: converts to grayscale (optional), resizes,
    and optionally inverts if it's detected as dark-on-light (to make it light-on-dark).
    Output is uint8 [0,255].
    """
    if grayscale and image.ndim == 3 and image.shape[2] == 3:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif grayscale and image.ndim == 3 and image.shape[2] == 1:
        processed_image = image.squeeze(axis=2)
    else:  # Already grayscale or not converting
        processed_image = image.copy()

    if invert_if_dark_on_light and grayscale:
        # Simple check: if average intensity is high, assume light background
        # This is a heuristic and might not always be correct.
        if np.mean(processed_image) > threshold_for_inversion_check:
            processed_image = 255 - processed_image  # Invert

    # Resize
    processed_image = cv2.resize(processed_image, target_size, interpolation=cv2.INTER_AREA)

    # Ensure uint8
    if processed_image.dtype != np.uint8:
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

    return processed_image


if __name__ == "__main__":
    # --- Test an image (create a dummy one) ---
    dummy_cv_bgr = np.zeros((30, 40, 3), dtype=np.uint8)
    cv2.putText(dummy_cv_bgr, "Test", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Green text

    dummy_cv_gray = cv2.cvtColor(dummy_cv_bgr, cv2.COLOR_BGR2GRAY)

    print(f"Original CV Gray shape: {dummy_cv_gray.shape}")

    # Test Cv -> PIL -> Cv
    pil_from_cv = cv_to_pil(dummy_cv_gray)
    print(f"PIL from CV mode: {pil_from_cv.mode}, size: {pil_from_cv.size}")
    cv_from_pil = pil_to_cv(pil_from_cv)
    print(f"CV from PIL shape: {cv_from_pil.shape}, dtype: {cv_from_pil.dtype}")
    assert np.array_equal(dummy_cv_gray, cv_from_pil), "CV <-> PIL grayscale conversion failed"

    pil_from_cv_color = cv_to_pil(dummy_cv_bgr)
    print(f"PIL from CV (color) mode: {pil_from_cv_color.mode}, size: {pil_from_cv_color.size}")
    cv_from_pil_color = pil_to_cv(pil_from_cv_color)
    print(f"CV from PIL (color) shape: {cv_from_pil_color.shape}, dtype: {cv_from_pil_color.dtype}")
    assert np.array_equal(dummy_cv_bgr, cv_from_pil_color), "CV <-> PIL color conversion failed"

    # Test np_to_tensor and tensor_to_np_cv
    tensor_from_np = np_to_tensor(dummy_cv_gray)
    print(
        f"Tensor from np (gray) shape: {tensor_from_np.shape}, dtype: {tensor_from_np.dtype}, device: {tensor_from_np.device}")
    np_from_tensor = tensor_to_np_cv(tensor_from_np)
    print(f"NP from tensor (gray) shape: {np_from_tensor.shape}, dtype: {np_from_tensor.dtype}")
    # Note: float conversions might lead to tiny differences, so direct equality might fail. Check range.
    assert np_from_tensor.max() <= 255 and np_from_tensor.min() >= 0, "Tensor -> NP value range error"

    # Test DGO preprocessing
    cfg_instance = get_config()  # Load default config for testing
    dgo_tensor = preprocess_image_for_dgo(dummy_cv_gray,
                                          target_size=cfg_instance.data_management.target_image_size,
                                          grayscale=cfg_instance.data_management.grayscale_input)
    print(f"DGO preprocessed tensor shape: {dgo_tensor.shape}")
    assert dgo_tensor.shape == (1, 1 if cfg_instance.data_management.grayscale_input else 3,
                                cfg_instance.data_management.target_image_size[0],
                                cfg_instance.data_management.target_image_size[1])

    # Test standardize_image
    standard_img = standardize_image(dummy_cv_bgr, target_size=(28, 28), grayscale=True, invert_if_dark_on_light=True)
    print(f"Standardized image shape: {standard_img.shape}, mean: {np.mean(standard_img):.2f}")
    # cv2.imshow("Standardized", standard_img); cv2.waitKey(0); cv2.destroyAllWindows() # If you want to see it

    print("\nImage utils tests completed.")