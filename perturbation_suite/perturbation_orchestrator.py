# ultimate_morph_generator/perturbation_suite/perturbation_orchestrator.py
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional

from ..config import get_config, PerturbationSuiteConfig
from ..utilities.type_definitions import CvImage, PerturbationName, PerturbationParams
from .base_transforms import BasePerturbations
# from .stroke_engine.stroke_perturbations import StrokePerturbations # Future
# from .style_mixer import StylePerturbations # Future
from ..dgo_oracle.dgo_model_handler import DGOModelHandler  # For guided perturbations
from ..utilities.logging_config import setup_logging

logger = setup_logging()


class PerturbationOrchestrator:
    """
    Selects and applies sequences of perturbations to an image.
    Can incorporate DGO guidance or more advanced strategies (RL, EA) in future.
    """

    def __init__(self,
                 suite_cfg: PerturbationSuiteConfig,
                 target_image_size: Tuple[int, int],
                 dgo_handler: Optional[DGOModelHandler] = None):  # DGO handler for guided perturbations
        self.suite_cfg = suite_cfg
        self.base_perturber = BasePerturbations(suite_cfg, target_image_size)
        # self.stroke_perturber: Optional[StrokePerturbations] = None # Initialize if stroke engine enabled
        # self.style_perturber: Optional[StylePerturbations] = None # Initialize if style mixer enabled

        self.dgo_handler = dgo_handler  # For gradient-guided or DGO-informed parameter selection

        self.available_methods: List[str] = self._get_available_methods()

        # For adaptive strategies (placeholders)
        self.method_weights: Optional[Dict[str, float]] = None  # Weights for selecting methods
        self.param_adaptation_rules: Optional[Dict] = None

        logger.info(f"PerturbationOrchestrator initialized. Available base methods: {self.available_methods}")
        # Add stroke/style methods if they get initialized.

    def _get_available_methods(self) -> List[str]:
        """Determines available perturbation methods based on config."""
        methods = []
        # Iterate over fields in PerturbationSuiteConfig that are PerturbationMethodConfig instances
        for method_name, method_config_field in PerturbationSuiteConfig.model_fields.items():
            # Check if the field's type annotation is PerturbationMethodConfig
            # This is a bit indirect. A better way is to have a list of method names in config.
            # Or, check if the attribute on suite_cfg is an instance of PerturbationMethodConfig.
            method_config_instance = getattr(self.suite_cfg, method_name, None)
            if isinstance(method_config_instance,
                          type(get_config().perturbation_suite.local_pixel)):  # Check instance type
                if method_config_instance.enabled:
                    methods.append(method_name)

        # Add other types of perturbers if enabled
        # if self.suite_cfg.stroke_engine_perturbations.enabled: methods.append("stroke_engine")
        # if self.suite_cfg.style_mixer.enabled: methods.append("style_mixer")
        return methods

    def select_perturbation_method(self) -> Optional[str]:
        """
        Selects a perturbation method to apply based on configured probabilities or adaptive weights.
        """
        if not self.available_methods:
            return None

        if self.method_weights:  # Use adaptive weights if available
            methods, weights = zip(*self.method_weights.items())
            return random.choices(methods, weights=weights, k=1)[0]
        else:  # Use configured probabilities
            # This logic needs careful implementation.
            # Each method_cfg has a 'probability_of_application'.
            # We need to choose one method from the available ones.
            # This is not a direct "sum of probabilities" choice.
            # More like: for each method, flip a coin with its prob. If it lands, apply.
            # Or, normalize these probabilities to sum to 1 and pick one.
            # For now, let's do a weighted random choice based on their 'probability_of_application'
            # acting as relative weights.

            method_names = []
            relative_weights = []
            for method_name in self.available_methods:
                method_cfg = getattr(self.suite_cfg, method_name)
                if method_cfg.enabled:  # Redundant check, _get_available_methods should handle
                    method_names.append(method_name)
                    # Use probability_of_application as a weight.
                    # If these are true probabilities (0-1), they might not sum to 1.
                    # If sum is > 1, random.choices normalizes. If sum is < 1, it still works.
                    relative_weights.append(method_cfg.probability_of_application)

            if not method_names: return None

            # Ensure weights are positive for random.choices
            # If all probabilities are 0, this will fail.
            if all(w <= 0 for w in relative_weights):
                # Fallback: uniform choice if all configured probabilities are zero or negative
                return random.choice(method_names)

            # Filter out methods with zero or negative probability before choices
            valid_choices = [(m, w) for m, w in zip(method_names, relative_weights) if w > 0]
            if not valid_choices:
                return random.choice(method_names)  # Fallback if all positive weights filtered out

            chosen_method = random.choices([vc[0] for vc in valid_choices],
                                           weights=[vc[1] for vc in valid_choices], k=1)[0]
            return chosen_method

    def apply_single_perturbation(self, image: CvImage,
                                  method_name: Optional[str] = None
                                  ) -> Tuple[CvImage, PerturbationName, PerturbationParams]:
        """
        Applies a single, randomly chosen (or specified) perturbation.
        Returns: (perturbed_image, applied_method_name, applied_params)
        """
        if method_name is None:
            method_name = self.select_perturbation_method()

        if not method_name:
            logger.debug("No perturbation method selected or available. Returning original image.")
            return image, "none", {}

        logger.debug(f"Orchestrator selected method: {method_name}")

        # Dispatch to the appropriate perturber
        if hasattr(self.base_perturber, 'apply_perturbation') and \
                method_name in PerturbationSuiteConfig.model_fields and \
                isinstance(getattr(self.suite_cfg, method_name, None),
                           type(get_config().perturbation_suite.local_pixel)):
            # It's a base perturbation method
            perturbed_image, params = self.base_perturber.apply_perturbation(image, method_name)
            return perturbed_image, method_name, params
        # elif method_name == "stroke_engine" and self.stroke_perturber:
        #     # perturbed_image, params = self.stroke_perturber.apply_random_stroke_perturbation(image)
        #     # return perturbed_image, method_name, params
        #     pass # Placeholder
        # elif method_name == "style_mixer" and self.style_perturber:
        #     # ...
        #     pass # Placeholder
        else:
            logger.warning(
                f"Method '{method_name}' selected by orchestrator, but no handler found or not a base method. Returning original.")
            return image, "none", {}

    def apply_perturbation_sequence(self, image: CvImage,
                                    num_perturbations: int = 1,
                                    enforce_distinct_methods: bool = False
                                    ) -> Tuple[CvImage, List[Tuple[PerturbationName, PerturbationParams]]]:
        """
        Applies a sequence of `num_perturbations` to the image.
        Returns the final perturbed image and a list of applied (method_name, params) tuples.
        """
        current_image = image.copy()
        applied_sequence_info = []

        methods_used_in_sequence = set()

        for i in range(num_perturbations):
            selected_method: Optional[str] = None
            if enforce_distinct_methods:
                # Try to pick a method not yet used in this sequence
                available_for_sequence = [m for m in self.available_methods if m not in methods_used_in_sequence]
                if not available_for_sequence:  # All methods used, allow repeats or stop
                    if len(self.available_methods) > 0:  # if there are methods at all
                        logger.debug(
                            "All available distinct methods used in sequence. Allowing repeats for remaining steps.")
                        selected_method = self.select_perturbation_method()
                    else:  # no methods available at all
                        break  # Stop if no methods to choose from
                else:
                    # Need to adapt select_perturbation_method to pick from a subset
                    # For now, simple random choice from available_for_sequence
                    # This bypasses configured probabilities if enforce_distinct_methods is True.
                    # A better way would be to re-normalize probabilities for the subset.
                    if available_for_sequence:
                        selected_method = random.choice(available_for_sequence)
            else:
                selected_method = self.select_perturbation_method()

            if not selected_method:
                logger.debug(f"Sequence step {i + 1}: No method selected. Ending sequence early.")
                break

            perturbed_image, method_name, params = self.apply_single_perturbation(current_image,
                                                                                  method_name=selected_method)

            if method_name != "none":  # If a perturbation was actually applied
                current_image = perturbed_image
                applied_sequence_info.append((method_name, params))
                methods_used_in_sequence.add(method_name)
            else:  # No perturbation applied (e.g. method not enabled, error, or p=0 for compose)
                logger.debug(f"Sequence step {i + 1}: Method '{selected_method}' resulted in no change.")

        return current_image, applied_sequence_info

    def apply_dgo_guided_perturbation(self, image: CvImage, target_char_idx: int,
                                      misguide_to_idx: Optional[int] = None,
                                      strength: float = 0.01,
                                      method: str = "fgsm_like") -> Tuple[
        CvImage, PerturbationName, PerturbationParams]:
        """
        Applies a perturbation guided by DGO gradients.
        Method:
            - "fgsm_like": Fast Gradient Sign Method style perturbation.
            - "gradient_ascent_logit": Move along raw gradient of target/misguide logit.
        Strength controls the epsilon for FGSM or step size for gradient ascent.
        """
        if not self.dgo_handler:
            logger.warning("DGO handler not available for guided perturbation. Returning original image.")
            return image, "none_dgo_guided", {}

        original_image_np = image.astype(np.float32) / 255.0  # Normalize to [0,1] for gradient calculation if needed

        # Get gradient from DGO
        # Loss type for gradient can be configured. For FGSM, usually CE loss w.r.t. a label.
        # Or, gradient of a specific logit.
        grad_loss_type = "logit_target_class"  # Example, make this configurable

        grad_np = self.dgo_handler.get_gradient_wrt_input(
            original_image_np,  # Pass [0,1] float image
            target_class_idx=target_char_idx if misguide_to_idx is None else None,
            misguide_to_class_idx=misguide_to_idx,
            loss_type=grad_loss_type
        )
        # grad_np is (H,W) for grayscale, or (C,H,W) for color, matching input tensor structure pre-squeeze.

        perturbed_image_np: np.ndarray
        if method == "fgsm_like":
            # FGSM: image_adv = image + strength * sign(gradient)
            # Note: If loss was NLL(output, true_label), gradient points to decrease loss.
            # For attack (increase loss or misclassify), sign depends on objective.
            # If misguide_to_idx is set, gradient points to make it *more like* misguide_to_idx (if loss was -logit(misguide)).
            # If target_char_idx is set, and loss was logit(target), gradient points to *increase* target.
            # If we want to make it *less* like target_char_idx, use -gradient or change loss definition.
            # Assuming gradient points in desired direction of change:
            signed_grad = np.sign(grad_np)

            # Reshape signed_grad if original_image_np is (H,W) and signed_grad is (H,W)
            # If original is (H,W,C) and grad_np is (C,H,W), transpose grad_np
            if original_image_np.ndim == 2 and signed_grad.ndim == 2:  # Grayscale
                perturbed_image_np = original_image_np + strength * signed_grad
            elif original_image_np.ndim == 3 and signed_grad.ndim == 3 and original_image_np.shape[2] == \
                    signed_grad.shape[0]:  # Color (H,W,C) vs (C,H,W)
                perturbed_image_np = original_image_np + strength * signed_grad.transpose(1, 2, 0)
            else:  # Mismatch
                logger.error(f"Shape mismatch for FGSM: image {original_image_np.shape}, grad {grad_np.shape}")
                return image, "failed_dgo_guided_fgsm", {}

        elif method == "gradient_ascent_logit":
            # image_adv = image + strength * gradient (normalize gradient?)
            # grad_norm = np.linalg.norm(grad_np)
            # normalized_grad = grad_np / (grad_norm + 1e-9) if grad_norm > 0 else grad_np
            # perturbed_image_np = original_image_np + strength * normalized_grad
            # Simpler: step along raw gradient
            if original_image_np.ndim == 2 and grad_np.ndim == 2:  # Grayscale
                perturbed_image_np = original_image_np + strength * grad_np
            elif original_image_np.ndim == 3 and grad_np.ndim == 3 and original_image_np.shape[2] == grad_np.shape[
                0]:  # Color
                perturbed_image_np = original_image_np + strength * grad_np.transpose(1, 2, 0)
            else:
                logger.error(f"Shape mismatch for GradAscent: image {original_image_np.shape}, grad {grad_np.shape}")
                return image, "failed_dgo_guided_gradascent", {}
        else:
            logger.warning(f"Unsupported DGO-guided method: {method}. Returning original.")
            return image, "none_dgo_guided", {}

        # Clip to [0,1] and convert back to uint8 [0,255]
        perturbed_image_np = np.clip(perturbed_image_np, 0.0, 1.0)
        final_perturbed_cvimage = (perturbed_image_np * 255.0).astype(np.uint8)

        params_used = {"dgo_guided_method": method, "strength": strength,
                       "target_idx_for_grad": target_char_idx, "misguide_idx_for_grad": misguide_to_idx,
                       "loss_type_for_grad": grad_loss_type}
        return final_perturbed_cvimage, f"dgo_guided_{method}", params_used

    # --- Placeholder for adaptive strategies ---
    def update_method_weights(self, feedback_data: Dict):
        """Update method selection weights based on feedback (e.g., DGO scores, novelty)."""
        # This would be part of an RL or bandit algorithm.
        logger.info("Adaptive method weight update called (placeholder).")

    def adapt_perturbation_parameters(self, method_name: str, feedback_data: Dict):
        """Adapt parameter ranges for a specific method based on feedback."""
        # E.g., if a method with certain params consistently fails structure check, narrow its ranges.
        logger.info(f"Adaptive parameter update for '{method_name}' called (placeholder).")


if __name__ == "__main__":
    # --- Test PerturbationOrchestrator ---
    from ..config import SystemConfig

    # Use the same config as BasePerturbations test for suite_cfg
    temp_sys_cfg_data_orch = {
        "perturbation_suite": {
            "local_pixel": {"enabled": True, "probability_of_application": 0.3},
            "elastic_deformation": {"enabled": True, "probability_of_application": 0.5},
            "fine_affine": {"enabled": True, "probability_of_application": 0.0},  # Test disabled
            "stroke_thickness_morph": {"enabled": True, "probability_of_application": 0.8}
        },
        "logging": {"level": "DEBUG"},
        # Add dummy DGO config for DGO-guided test
        "data_management": {"target_image_size": (28, 28), "grayscale_input": True},
        "dgo_oracle": {
            "model_architecture": "BaseCNN", "num_classes": 10,
            "pretrained_model_path": None, "feature_extraction_layer_name": "relu3",
        }
    }
    from ..config import _config_instance

    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_orch)
    cfg = get_config()

    # Create dummy DGO handler for guided test
    dgo_h_dummy = DGOModelHandler(cfg.dgo_oracle, cfg.data_management, cfg.get_actual_device())

    orchestrator = PerturbationOrchestrator(suite_cfg=cfg.perturbation_suite,
                                            target_image_size=cfg.data_management.target_image_size,
                                            dgo_handler=dgo_h_dummy)  # Pass dummy DGO

    dummy_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    cv2.putText(dummy_image, "O", (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200), 2)

    print("\n--- Testing PerturbationOrchestrator Selection ---")
    selected_counts = {m: 0 for m in orchestrator.available_methods}
    selected_counts["none"] = 0  # If no method chosen due to probabilities
    for _ in range(100):
        method = orchestrator.select_perturbation_method()
        if method:
            selected_counts[method] += 1
        else:
            selected_counts["none"] += 1  # Should be rare if probs are high
    print(f"Method selection counts (out of 100): {selected_counts}")
    assert "fine_affine" not in selected_counts or selected_counts.get("fine_affine", 0) == 0  # Since it was disabled

    print("\n--- Testing Single Perturbation Application ---")
    perturbed_img_single, method_applied, params_applied = orchestrator.apply_single_perturbation(dummy_image.copy())
    print(f"Applied single: {method_applied} with params: {params_applied}")
    if method_applied != "none":
        assert not np.array_equal(dummy_image, perturbed_img_single) or not params_applied
        # cv2.imshow("Single Perturbed", perturbed_img_single); cv2.waitKey(1)

    print("\n--- Testing Perturbation Sequence ---")
    num_in_seq = 3
    perturbed_img_seq, seq_info = orchestrator.apply_perturbation_sequence(dummy_image.copy(),
                                                                           num_perturbations=num_in_seq,
                                                                           enforce_distinct_methods=True)
    print(f"Applied sequence of {len(seq_info)} perturbations:")
    for name, params_ in seq_info:
        print(f"  - {name}: {params_}")
    assert len(seq_info) <= num_in_seq
    # cv2.imshow("Sequence Perturbed", perturbed_img_seq); cv2.waitKey(1)

    print("\n--- Testing DGO-Guided Perturbation ---")
    perturbed_img_dgo, method_dgo, params_dgo = orchestrator.apply_dgo_guided_perturbation(
        dummy_image.copy(), target_char_idx=3, strength=0.05, method="fgsm_like"
    )
    print(f"Applied DGO-guided: {method_dgo} with params: {params_dgo}")
    if method_dgo != "none_dgo_guided" and not method_dgo.startswith("failed"):
        assert not np.array_equal(dummy_image, perturbed_img_dgo)
        # cv2.imshow("DGO Guided Perturbed", perturbed_img_dgo); cv2.waitKey(1)

    # cv2.destroyAllWindows()
    print("\nPerturbationOrchestrator tests completed.")