# ultimate_morph_generator/perturbation_suite/style_mixer.py
import cv2
import numpy as np
import os
import random
from typing import List, Tuple, Dict, Any, Optional

from ...config import get_config, StyleMixerConfig
from ...utilities.type_definitions import CvImage
from ...utilities.logging_config import setup_logging

logger = setup_logging()


# --- Placeholder for a Deep Learning based Style Transfer Model ---
class DLStyleTransferModel:
    """
    Conceptual wrapper for a deep learning model that performs controlled style transfer.
    This would load a pretrained model (e.g., from a .pth file or a framework like TF Hub/PyTorch Hub).
    """

    def __init__(self, model_path: Optional[str], device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Load the actual PyTorch/TensorFlow model here
                # self.model = torch.load(self.model_path, map_location=self.device)
                # self.model.eval()
                logger.info(f"Conceptual DL Style Transfer model loaded from {self.model_path}")
                # For now, it's just a placeholder:
                self.model = lambda content_img, style_img, strength: content_img  # Dummy pass-through
            except Exception as e:
                logger.error(f"Failed to load DL Style Transfer model: {e}")
                self.model = lambda content_img, style_img, strength: content_img  # Dummy on error
        else:
            logger.warning("DL Style Transfer model path not provided or invalid. Using dummy pass-through.")
            self.model = lambda content_img, style_img, strength: content_img  # Dummy

    def transfer_style(self, content_image: CvImage, style_image: CvImage, strength: float) -> CvImage:
        """
        Applies the style of style_image to content_image with a given strength.
        Input and output are CvImage (NumPy arrays).
        """
        if self.model:
            # Preprocess content_image and style_image for the model
            # Run model inference
            # Postprocess output back to CvImage
            # The dummy model just returns the content image
            logger.debug(f"Applying conceptual DL style transfer (strength: {strength:.2f}). Using dummy pass-through.")
            # Example of a very simple blend if not using a real model:
            # if content_image.shape == style_image.shape:
            #    resized_style = style_image
            # else:
            #    resized_style = cv2.resize(style_image, (content_image.shape[1], content_image.shape[0]))
            # if content_image.ndim == 2 and resized_style.ndim == 3:
            #    resized_style = cv2.cvtColor(resized_style, cv2.COLOR_BGR2GRAY)
            # if content_image.ndim == 3 and resized_style.ndim == 2:
            #    resized_style = cv2.cvtColor(resized_style, cv2.COLOR_GRAY2BGR)
            #
            # return cv2.addWeighted(content_image, 1 - strength, resized_style, strength, 0)
            return self.model(content_image, style_image, strength)
        return content_image


class StyleMixer:
    """
    Applies subtle stylistic variations to character images.
    Can use simple image processing techniques or (conceptually) a DL model.
    """

    def __init__(self, style_cfg: StyleMixerConfig, target_image_size: Tuple[int, int]):
        self.cfg = style_cfg
        self.target_image_size = target_image_size  # H, W
        self.style_images_cache: List[CvImage] = []

        self.dl_style_model: Optional[DLStyleTransferModel] = None

        if self.cfg.enabled:
            self._load_style_sources()
            if getattr(self.cfg, 'use_deep_learning_model', False):  # Add this to StyleMixerConfig if desired
                dl_model_path = getattr(self.cfg, 'deep_learning_style_model_path', None)
                self.dl_style_model = DLStyleTransferModel(model_path=dl_model_path)

    def _load_style_sources(self):
        """Loads style reference images/textures from the configured directory."""
        if not self.cfg.style_source_dir or not os.path.exists(self.cfg.style_source_dir):
            logger.warning(
                f"Style source directory not provided or invalid: {self.cfg.style_source_dir}. Style mixer may have limited effect.")
            return

        for fname in os.listdir(self.cfg.style_source_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    path = os.path.join(self.cfg.style_source_dir, fname)
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load as is (color or gray)
                    if img is not None:
                        # Resize to target_image_size for consistency or specific model input
                        # Resizing style images might alter their stylistic properties.
                        # Or, use patches from larger style images.
                        # For now, let's resize.
                        resized_style_img = cv2.resize(img, (self.target_image_size[1], self.target_image_size[0]))
                        self.style_images_cache.append(resized_style_img)
                    else:
                        logger.warning(f"Could not load style image: {path}")
                except Exception as e:
                    logger.error(f"Error loading style image {fname}: {e}")
        logger.info(f"Loaded {len(self.style_images_cache)} style source images for StyleMixer.")

    def _apply_texture_overlay(self, image: CvImage, strength: float) -> CvImage:
        """
        Applies a subtle texture overlay using one of the cached style images.
        This is a very basic "style" effect.
        """
        if not self.style_images_cache:
            return image

        texture_img = random.choice(self.style_images_cache).copy()

        # Ensure texture_img matches image channels (grayscale/color)
        if image.ndim == 2 and texture_img.ndim == 3:  # Image is gray, texture is color
            texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and texture_img.ndim == 2:  # Image is color, texture is gray
            texture_img = cv2.cvtColor(texture_img, cv2.COLOR_GRAY2BGR)

        # Ensure same size (should be handled by _load_style_sources resize)
        if texture_img.shape[:2] != image.shape[:2]:
            texture_img = cv2.resize(texture_img, (image.shape[1], image.shape[0]))

        # Blend: image * (1-strength) + texture * strength
        # To make it look like texture *on* the character, we might need a mask of the character.
        # Simple overlay blend:
        try:
            # Convert to float for blending to avoid uint8 saturation issues with low strength
            image_float = image.astype(np.float32)
            texture_float = texture_img.astype(np.float32)

            blended = cv2.addWeighted(image_float, 1.0, texture_float, strength, 0)  # Texture on top
            # Or, if texture should modify the character, not just overlay:
            # blended = cv2.addWeighted(image_float, 1.0 - strength, texture_float, strength, 0)

            # Clip and convert back to uint8
            blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
            return blended_uint8
        except cv2.error as e:
            logger.error(
                f"OpenCV error during texture overlay: {e}. Image shape: {image.shape}, Texture shape: {texture_img.shape}")
            return image

    def _apply_pencil_sketch_effect(self, image: CvImage, strength: float) -> CvImage:
        """
        Simulates a very light pencil sketch effect.
        Strength could control kernel sizes or blending amount.
        """
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        # 1. Invert image
        img_invert = cv2.bitwise_not(gray_image)
        # 2. Blur inverted image
        # Kernel size can be tied to strength: smaller kernel for finer sketch
        k_size = max(3, int(7 * (1.0 - strength)))  # Example: strength 0->k=7, strength 1->k=3
        if k_size % 2 == 0: k_size += 1  # Must be odd
        img_blur = cv2.GaussianBlur(img_invert, (k_size, k_size), 0)
        # 3. Invert blurred image
        img_blur_inv = cv2.bitwise_not(img_blur)
        # 4. Dodge blend: (gray_image * 255) / (255 - img_blur_inv + 1)
        # This creates the sketch lines.
        sketch = cv2.divide(gray_image, 255 - img_blur_inv, scale=256.0)  # cv2.divide handles saturation

        # Convert back to 3 channels if original was color, and blend
        if image.ndim == 3:
            sketch_color = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            # Blend original color image with the sketch effect
            # Low strength means mostly original, high strength means more sketch
            # Final blending strength might be different from the 'strength' param used for kernel.
            blend_alpha = strength * 0.5  # How much of sketch to show
            output_image = cv2.addWeighted(image, 1 - blend_alpha, sketch_color, blend_alpha, 0)
            return output_image
        else:
            return sketch

    def apply_style_perturbation(self, image: CvImage) -> Tuple[CvImage, Dict[str, Any]]:
        """
        Applies a selected style perturbation.
        Returns the styled image and parameters of the applied style.
        """
        if not self.cfg.enabled:
            return image, {"style_method": "none", "reason": "disabled"}

        # Choose a style application method
        # For now, let's randomly pick between texture overlay and sketch, or DL if available.
        # This could be made more configurable.

        # Parameter: strength of style application
        strength = random.uniform(self.cfg.strength_range[0], self.cfg.strength_range[1])

        applied_method_name = "none"
        params_used = {"strength": strength}

        available_ops = []
        if self.dl_style_model and self.style_images_cache:  # DL model needs a style image
            available_ops.append("deep_learning_transfer")
        if self.style_images_cache:  # Texture overlay needs style images
            available_ops.append("texture_overlay")
        available_ops.append("pencil_sketch_effect")  # Sketch doesn't need style images

        if not available_ops:
            logger.debug("No style operations available (e.g., no style images loaded).")
            return image, {"style_method": "none", "reason": "no_ops_available"}

        chosen_op = random.choice(available_ops)

        output_image = image.copy()

        if chosen_op == "deep_learning_transfer" and self.dl_style_model and self.style_images_cache:
            style_ref_img = random.choice(self.style_images_cache)
            output_image = self.dl_style_model.transfer_style(image, style_ref_img, strength)
            applied_method_name = "dl_style_transfer"
            params_used["style_reference_image_sample"] = "random_from_cache"  # Can log specific style image if needed

        elif chosen_op == "texture_overlay":
            output_image = self._apply_texture_overlay(image, strength)
            applied_method_name = "texture_overlay"

        elif chosen_op == "pencil_sketch_effect":
            output_image = self._apply_pencil_sketch_effect(image, strength)
            applied_method_name = "pencil_sketch"

        else:  # Should not happen if available_ops is managed correctly
            logger.warning(f"StyleMixer: Chosen op '{chosen_op}' not handled.")
            return image, {"style_method": "error_op_not_handled"}

        logger.debug(f"Applied style: {applied_method_name} with strength {strength:.2f}")
        return output_image, {"style_method": applied_method_name, "params": params_used}


if __name__ == "__main__":
    # --- Test StyleMixer ---
    from ....config import SystemConfig  # Adjust relative import

    # Create dummy style source directory and images
    dummy_style_dir = "./temp_style_sources/"
    os.makedirs(dummy_style_dir, exist_ok=True)
    style_img1 = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)  # A color texture
    cv2.imwrite(os.path.join(dummy_style_dir, "style1.png"), style_img1)
    style_img2 = np.random.randint(50, 150, (64, 64), dtype=np.uint8)  # A grayscale texture
    cv2.imwrite(os.path.join(dummy_style_dir, "style2.png"), style_img2)

    temp_sys_cfg_data_style = {
        "perturbation_suite": {
            "style_mixer": {  # This is StyleMixerConfig
                "enabled": True,
                "style_source_dir": dummy_style_dir,
                "strength_range": (0.1, 0.4),
                # "use_deep_learning_model": False, # Default
                # "deep_learning_style_model_path": None
            }
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_style)
    cfg_glob = get_config()
    style_mixer_config = cfg_glob.perturbation_suite.style_mixer

    mixer = StyleMixer(style_cfg=style_mixer_config, target_image_size=(32, 32))

    # Create a test character image
    test_char_img = np.zeros((32, 32), dtype=np.uint8)
    cv2.putText(test_char_img, "S", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220), 2)

    # cv2.imshow("Original Char for StyleMixer", test_char_img); cv2.waitKey(1)

    print("\n--- Testing StyleMixer Application ---")
    styled_image, style_info = mixer.apply_style_perturbation(test_char_img.copy())

    print(f"Applied style info: {style_info}")
    print(f"Original char shape: {test_char_img.shape}, Styled char shape: {styled_image.shape}")
    assert styled_image.shape[:2] == test_char_img.shape[:2]  # Allow for color change

    # Visual check
    if style_info.get("style_method", "none") != "none":
        # combined_style = np.hstack((cv2.cvtColor(test_char_img, cv2.COLOR_GRAY2BGR) if test_char_img.ndim==2 else test_char_img,
        #                             cv2.cvtColor(styled_image, cv2.COLOR_GRAY2BGR) if styled_image.ndim==2 else styled_image))
        # cv2.imshow(f"Original vs Styled ({style_info.get('style_method')})", combined_style)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pass
    print("StyleMixer test completed (visual check recommended).")

    # Clean up dummy style dir
    import shutil

    shutil.rmtree(dummy_style_dir, ignore_errors=True)