# ultimate_morph_generator/perturbation_suite/base_transforms.py
import cv2
import numpy as np
import random
import albumentations as A  # type: ignore
from albumentations.core.transforms_interface import ImageOnlyTransform
from typing import Tuple, List, Dict, Any, Union, Optional, Callable

from ..config import get_config, PerturbationMethodConfig, PerturbationSuiteConfig
from ..utilities.type_definitions import CvImage
from ..utilities.logging_config import setup_logging

logger = setup_logging()


# --- Custom Albumentations Transform for Local Pixel Perturbation (Example) ---
class LocalPixelShuffle(ImageOnlyTransform):
    """
    Albumentations transform to shuffle pixels within small local neighborhoods.
    This is a more controlled version of random pixel noise.
    """

    def __init__(self, neighborhood_size: int = 3, perturb_density: float = 0.05,
                 always_apply: bool = False, p: float = 0.5):
        """
        Args:
            neighborhood_size (int): Size of the square neighborhood (e.g., 3 for 3x3). Must be odd.
            perturb_density (float): Fraction of pixels to select as centers for shuffling.
            p (float): Probability of applying the transform.
        """
        super(LocalPixelShuffle, self).__init__(always_apply, p)
        if neighborhood_size % 2 == 0:
            raise ValueError("Neighborhood size must be odd.")
        self.neighborhood_size = neighborhood_size
        self.perturb_density = perturb_density

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img_perturbed = img.copy()
        h, w = img.shape[:2]
        is_color = img.ndim == 3 and img.shape[2] > 1

        num_centers = int(h * w * self.perturb_density)
        hs = self.neighborhood_size // 2  # half_size

        for _ in range(num_centers):
            # Select a center pixel
            cx, cy = random.randint(hs, w - 1 - hs), random.randint(hs, h - 1 - hs)

            # Define neighborhood boundaries
            y_min, y_max = cy - hs, cy + hs + 1
            x_min, x_max = cx - hs, cx + hs + 1

            neighborhood = img[y_min:y_max, x_min:x_max].copy()  # Get a copy of the neighborhood

            # Shuffle pixels within this neighborhood
            if is_color:
                # Shuffle along H and W dimensions for each channel independently or together
                # Simpler: shuffle flattened neighborhood pixels then reshape
                original_shape = neighborhood.shape
                flat_neighborhood = neighborhood.reshape(-1, original_shape[2])  # (N_pixels_in_hood, C)
                np.random.shuffle(flat_neighborhood)  # Shuffles rows (pixels)
                shuffled_neighborhood = flat_neighborhood.reshape(original_shape)
            else:  # Grayscale
                original_shape = neighborhood.shape
                flat_neighborhood = neighborhood.flatten()
                np.random.shuffle(flat_neighborhood)
                shuffled_neighborhood = flat_neighborhood.reshape(original_shape)

            img_perturbed[y_min:y_max, x_min:x_max] = shuffled_neighborhood

        return img_perturbed

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("neighborhood_size", "perturb_density")


class BasePerturbations:
    """
    Applies various basic image perturbations using Albumentations.
    Each method is parameterizable based on the global configuration.
    """

    def __init__(self, suite_cfg: PerturbationSuiteConfig, target_image_size: Tuple[int, int]):
        self.suite_cfg = suite_cfg
        self.target_image_size = target_image_size  # H, W
        self._transforms_cache: Dict[str, Callable] = {}  # Cache composed transforms

    def _get_params_from_config(self, method_cfg: PerturbationMethodConfig,
                                param_name: str, default_value: Any) -> Any:
        """Helper to get a specific parameter value, choosing randomly if range is given."""
        param_spec = method_cfg.param_ranges.get(param_name)
        if param_spec is None:
            return default_value

        if isinstance(param_spec, list):  # Choose one from list
            return random.choice(param_spec)
        elif isinstance(param_spec, tuple) and len(param_spec) == 2:
            low, high = param_spec
            if isinstance(low, int) and isinstance(high, int):
                return random.randint(low, high)
            elif isinstance(low, float) and isinstance(high, float):
                return random.uniform(low, high)
            else:  # Mixed types or other tuple format not directly supported by uniform/randint
                logger.warning(
                    f"Parameter '{param_name}' has tuple spec {param_spec} not directly usable. Returning default.")
                return default_value
        else:  # Single value specified directly (not a range/list)
            return param_spec

    def _create_or_get_transform(self, method_name: str, method_cfg: PerturbationMethodConfig) -> Optional[A.Compose]:
        """Creates or retrieves a cached Albumentations Compose object for the given method."""
        if method_name in self._transforms_cache and False:  # Caching can be tricky if params need to be dynamic per call
            # For now, re-create each time to ensure random params are fresh, unless we design for fixed param sets.
            # return self._transforms_cache[method_name]
            pass

        transform_list = []
        prob_application = method_cfg.probability_of_application  # This 'p' is for the whole composed transform
        # Individual A.transforms also have 'p'

        if method_name == "local_pixel" and method_cfg.enabled:
            # Uses custom LocalPixelShuffle
            nh_size = self._get_params_from_config(method_cfg, "neighborhood_size", 3)
            p_density = self._get_params_from_config(method_cfg, "perturb_density", 0.05)
            # intensity_noise_range for adding noise, not shuffle. Shuffle is different.
            # For intensity noise, could use A.GaussNoise or custom.
            # Here, local_pixel refers to shuffling.
            transform_list.append(LocalPixelShuffle(neighborhood_size=nh_size, perturb_density=p_density,
                                                    p=1.0))  # Apply if method is chosen

        elif method_name == "elastic_deformation" and method_cfg.enabled:
            alpha = self._get_params_from_config(method_cfg, "alpha", 30.0)
            sigma = self._get_params_from_config(method_cfg, "sigma", 4.0)
            alpha_affine = self._get_params_from_config(method_cfg, "alpha_affine", 5.0)
            # Albumentations ElasticTransform: alpha is displacement scale, sigma is smoothness.
            # alpha_affine controls affine part.
            transform_list.append(A.ElasticTransform(
                alpha=alpha, sigma=sigma, alpha_affine=alpha_affine,
                p=1.0,  # Apply if method chosen
                border_mode=cv2.BORDER_CONSTANT,  # Or BORDER_REPLICATE, etc.
                value=0  # Fill value for border (black for char on white bg, or vice-versa)
            ))

        elif method_name == "fine_affine" and method_cfg.enabled:
            # Albumentations Affine takes ranges for rotation, scale, shear
            max_rot = self._get_params_from_config(method_cfg, "max_rotation_degrees", 5.0)
            scale_delta = self._get_params_from_config(method_cfg, "max_scale_delta", 0.1)
            shear_x_deg = self._get_params_from_config(method_cfg, "max_shear_degrees_x", 3.0)
            shear_y_deg = self._get_params_from_config(method_cfg, "max_shear_degrees_y", 3.0)
            trans_x_pct = self._get_params_from_config(method_cfg, "translate_percent_x", (-0.03, 0.03))
            trans_y_pct = self._get_params_from_config(method_cfg, "translate_percent_y", (-0.03, 0.03))

            transform_list.append(A.Affine(
                rotate=(-max_rot, max_rot),
                scale=(1.0 - scale_delta, 1.0 + scale_delta),
                shear={'x': (-shear_x_deg, shear_x_deg), 'y': (-shear_y_deg, shear_y_deg)},
                translate_percent={'x': trans_x_pct, 'y': trans_y_pct},
                p=1.0,  # Apply if method chosen
                mode=cv2.BORDER_CONSTANT, cval=0
            ))

        elif method_name == "stroke_thickness_morph" and method_cfg.enabled:
            op_type = self._get_params_from_config(method_cfg, "operation_type", "dilate")
            k_size = self._get_params_from_config(method_cfg, "kernel_size", 3)

            # Albumentations doesn't have direct dilate/erode for general images in main API
            # as these are often very specific. We can use OpenCV directly within a custom transform
            # or use `A.Lambda` to wrap cv2 functions.
            # For simplicity, let's make a small custom transform for this.
            class Morphological(ImageOnlyTransform):
                def __init__(self, operation: str, kernel_s: int, always_apply=False, p=0.5):
                    super().__init__(always_apply, p)
                    self.operation = operation
                    self.kernel = np.ones((kernel_s, kernel_s), np.uint8)

                def apply(self, img, **params):
                    if self.operation == "dilate":
                        return cv2.dilate(img, self.kernel, iterations=1)
                    elif self.operation == "erode":
                        return cv2.erode(img, self.kernel, iterations=1)
                    return img  # Should not happen

                def get_transform_init_args_names(self):
                    return ("operation", "kernel_s")

            transform_list.append(Morphological(operation=op_type, kernel_s=k_size, p=1.0))

        # Add other common perturbations from Albumentations if desired:
        # e.g., A.GaussNoise, A.MotionBlur, A.RandomBrightnessContrast, A.Perspective
        # Example:
        # if method_name == "gaussian_noise" and method_cfg.enabled:
        #    var_limit = self._get_params_from_config(method_cfg, "var_limit", (10.0, 50.0))
        #    transform_list.append(A.GaussNoise(var_limit=var_limit, mean=0, p=1.0))

        if not transform_list:
            return None  # Method not recognized or not enabled or no transforms defined

        # The probability `prob_application` is for the entire Compose.
        # Individual transforms inside are set to p=1.0 because the Compose itself handles the probability.
        composed_transform = A.Compose(transform_list, p=prob_application)
        # self._transforms_cache[method_name] = composed_transform # Cache if re-using fixed params
        return composed_transform

    def apply_perturbation(self, image: CvImage, method_name: str) -> Tuple[CvImage, Dict[str, Any]]:
        """
        Applies a named perturbation method to the image.
        The parameters for the perturbation are drawn randomly based on config for that method.
        Returns the perturbed image and the actual parameters used.
        """
        method_cfg = getattr(self.suite_cfg, method_name, None)
        if not method_cfg or not isinstance(method_cfg, PerturbationMethodConfig) or not method_cfg.enabled:
            logger.debug(f"Perturbation method '{method_name}' not configured, not enabled, or invalid. Skipping.")
            return image, {}  # Return original image and no params

        # Create transform with freshly sampled parameters for this call
        # This means _create_or_get_transform will effectively re-create it.
        # The 'p' in A.Compose is set to the method_cfg.probability_of_application.
        # So, the composed transform itself might not apply. We need to control this.

        # If we want to *guarantee* application when this function is called for a *specific selected method*:
        # We should use p=1 for the A.Compose here, and handle method selection probability *outside* this function.
        # The PerturbationOrchestrator will decide *which* method to apply based on its own logic
        # (which can use method_cfg.probability_of_application).

        # Let's assume the orchestrator has already chosen this method. So, apply it (p=1).
        # Create a temporary Compose with p=1 for the selected method's transforms.
        # The internal parameters of the transforms are still randomized per call to _get_params_from_config.

        temp_transform_list = []
        actual_params_used = {}  # Store actual parameters chosen for this run

        # Re-implementing parts of _create_or_get_transform logic here to ensure p=1 for application
        # and to capture actual_params_used.
        if method_name == "local_pixel":
            nh_size = self._get_params_from_config(method_cfg, "neighborhood_size", 3)
            p_density = self._get_params_from_config(method_cfg, "perturb_density", 0.05)
            temp_transform_list.append(LocalPixelShuffle(neighborhood_size=nh_size, perturb_density=p_density, p=1.0))
            actual_params_used = {"neighborhood_size": nh_size, "perturb_density": p_density}

        elif method_name == "elastic_deformation":
            alpha = self._get_params_from_config(method_cfg, "alpha", 30.0)
            sigma = self._get_params_from_config(method_cfg, "sigma", 4.0)
            alpha_affine = self._get_params_from_config(method_cfg, "alpha_affine", 5.0)
            temp_transform_list.append(A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=1.0,
                                                          border_mode=cv2.BORDER_CONSTANT, value=0))
            actual_params_used = {"alpha": alpha, "sigma": sigma, "alpha_affine": alpha_affine}

        elif method_name == "fine_affine":
            max_rot = self._get_params_from_config(method_cfg, "max_rotation_degrees", 5.0)
            scale_delta = self._get_params_from_config(method_cfg, "max_scale_delta", 0.1)
            shear_x_deg = self._get_params_from_config(method_cfg, "max_shear_degrees_x", 3.0)
            shear_y_deg = self._get_params_from_config(method_cfg, "max_shear_degrees_y", 3.0)
            trans_x_pct = self._get_params_from_config(method_cfg, "translate_percent_x", (-0.03, 0.03))  # Tuple
            trans_y_pct = self._get_params_from_config(method_cfg, "translate_percent_y", (-0.03, 0.03))  # Tuple

            temp_transform_list.append(A.Affine(
                rotate=(-max_rot, max_rot), scale=(1.0 - scale_delta, 1.0 + scale_delta),
                shear={'x': (-shear_x_deg, shear_x_deg), 'y': (-shear_y_deg, shear_y_deg)},
                translate_percent={'x': trans_x_pct, 'y': trans_y_pct},  # Pass tuple directly
                p=1.0, mode=cv2.BORDER_CONSTANT, cval=0
            ))
            actual_params_used = {"max_rotation_degrees": max_rot, "max_scale_delta": scale_delta,
                                  "max_shear_degrees_x": shear_x_deg, "max_shear_degrees_y": shear_y_deg,
                                  "translate_percent_x": trans_x_pct, "translate_percent_y": trans_y_pct}

        elif method_name == "stroke_thickness_morph":
            op_type = self._get_params_from_config(method_cfg, "operation_type", "dilate")
            k_size = self._get_params_from_config(method_cfg, "kernel_size", 3)

            class Morphological(ImageOnlyTransform):  # Redefine locally or make it accessible
                def __init__(self, operation: str, kernel_s: int, always_apply=False, p=0.5):
                    super().__init__(always_apply, p)
                    self.operation = operation
                    self.kernel = np.ones((kernel_s, kernel_s), np.uint8)

                def apply(self, img, **params):
                    if self.operation == "dilate":
                        return cv2.dilate(img, self.kernel, iterations=1)
                    elif self.operation == "erode":
                        return cv2.erode(img, self.kernel, iterations=1)
                    return img

                def get_transform_init_args_names(self):
                    return ("operation", "kernel_s")

            temp_transform_list.append(Morphological(operation=op_type, kernel_s=k_size, p=1.0))
            actual_params_used = {"operation_type": op_type, "kernel_size": k_size}

        else:
            logger.warning(
                f"No specific transform logic defined for method '{method_name}' in apply_perturbation. Returning original.")
            return image, {}

        if not temp_transform_list:
            return image, {}

        # Compose with p=1.0 to ensure it runs if this method was selected.
        final_transform = A.Compose(temp_transform_list, p=1.0)

        try:
            # Albumentations expects color images (H,W,C) by default for many transforms.
            # If grayscale (H,W), need to handle potential issues.
            # Some transforms work fine, others might need img to be (H,W,1) or converted.
            # For simplicity, ensure image is in a compatible format.
            is_grayscale_input = image.ndim == 2
            temp_image_for_aug = image
            if is_grayscale_input:
                # Many Albumentations transforms prefer (H,W,C) or work better with it.
                # Convert to 3-channel grayscale for augmentation, then convert back if needed.
                # Or, ensure custom transforms handle 2D arrays correctly.
                # LocalPixelShuffle and Morphological are written to handle 2D.
                # ElasticTransform and Affine usually handle grayscale if channels=1.
                # Let's assume current transforms are okay with (H,W) or (H,W,1).
                pass

            perturbed_result = final_transform(image=temp_image_for_aug)
            perturbed_image = perturbed_result['image']

            return perturbed_image, actual_params_used

        except Exception as e:
            logger.error(f"Error applying perturbation '{method_name}': {e}", exc_info=True)
            return image, {}  # Return original on error


if __name__ == "__main__":
    # --- Test BasePerturbations ---
    from ..config import SystemConfig  # Full system config for testing

    temp_sys_cfg_data = {
        "perturbation_suite": {  # Uses default PerturbationSuiteConfig structure
            "local_pixel": {"enabled": True, "probability_of_application": 0.8,
                            "param_ranges": {"neighborhood_size": [3], "perturb_density": (0.05, 0.1)}},
            "elastic_deformation": {"enabled": True, "probability_of_application": 0.7},
            "fine_affine": {"enabled": True, "probability_of_application": 0.9},
            "stroke_thickness_morph": {"enabled": True, "probability_of_application": 0.6}
        },
        "logging": {"level": "DEBUG"}
    }
    # This is a partial config for testing PerturbationSuiteConfig directly
    # Full SystemConfig would be needed if other parts are accessed by BasePerturbations.
    # For BasePerturbations, it primarily needs its own suite_cfg.
    # Let's construct PerturbationSuiteConfig directly for focused test.

    test_pert_suite_cfg = PerturbationSuiteConfig.model_validate(temp_sys_cfg_data["perturbation_suite"])

    perturber = BasePerturbations(suite_cfg=test_pert_suite_cfg, target_image_size=(32, 32))

    dummy_image_gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    cv2.putText(dummy_image_gray, "P", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128), 2)

    dummy_image_color = cv2.cvtColor(dummy_image_gray, cv2.COLOR_GRAY2BGR)  # For testing color path

    print("\n--- Testing BasePerturbations Applicator ---")

    test_methods = ["local_pixel", "elastic_deformation", "fine_affine", "stroke_thickness_morph"]
    for method in test_methods:
        print(f"\nTesting method: {method}")
        perturbed_img_gray, params_gray = perturber.apply_perturbation(dummy_image_gray.copy(), method)
        perturbed_img_color, params_color = perturber.apply_perturbation(dummy_image_color.copy(), method)

        print(f"  Params used (gray): {params_gray}")
        print(f"  Gray - Original shape: {dummy_image_gray.shape}, Perturbed shape: {perturbed_img_gray.shape}")
        assert perturbed_img_gray.shape == dummy_image_gray.shape
        # Check if image changed (it should, unless params were trivial)
        # assert not np.array_equal(dummy_image_gray, perturbed_img_gray) or not params_gray # can fail if params lead to no change

        print(f"  Params used (color): {params_color}")
        print(f"  Color - Original shape: {dummy_image_color.shape}, Perturbed shape: {perturbed_img_color.shape}")
        assert perturbed_img_color.shape == dummy_image_color.shape

        # Optional: Display images
        # combined_gray = np.hstack((dummy_image_gray, perturbed_img_gray))
        # cv2.imshow(f"Original vs Perturbed (Gray) - {method}", combined_gray)
        # combined_color = np.hstack((dummy_image_color, perturbed_img_color))
        # cv2.imshow(f"Original vs Perturbed (Color) - {method}", combined_color)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #    break

    # cv2.destroyAllWindows()
    print("\nBasePerturbations tests completed (visual check recommended if display enabled).")