# ultimate_morph_generator/dgo_oracle/dgo_model_handler.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union, Type
from contextlib import contextmanager

from ..config import get_config, DGOOracleConfig, DataManagementConfig
from ..utilities.image_utils import preprocess_image_for_dgo, \
    np_to_tensor  # Assuming np_to_tensor handles (H,W) or (H,W,C)
from ..utilities.type_definitions import CvImage, PILImage, DGOOutput, FeatureVector, DGOUncertainty, DGOScores
from ..utilities.logging_config import setup_logging

# Import model architectures
from .architectures.base_cnn import BaseCNN
from .architectures.resnet_variant import ResNetForChars

# from .architectures.vit_small import ViTSmall # If implemented

logger = setup_logging()

MODEL_REGISTRY = {
    "BaseCNN": BaseCNN,
    "ResNetVariant": ResNetForChars.resnet18_char_variant,  # Static method for specific variant
    # "ViTSmall": ViTSmall,
}


class DGOModelHandler:
    """
    Manages a DGO model (PyTorch nn.Module), providing interfaces for:
    - Loading and saving the model.
    - Performing predictions (classification).
    - Extracting feature vectors.
    - Calculating gradients of outputs w.r.t. inputs (for adversarial guidance).
    - Estimating prediction uncertainty (e.g., MC Dropout).
    """

    def __init__(self, dgo_config: DGOOracleConfig, data_config: DataManagementConfig,
                 device: Optional[torch.device] = None):
        self.dgo_cfg = dgo_config
        self.data_cfg = data_config
        self.device = device if device else get_config().get_actual_device()  # Use global config's device logic

        self.model: nn.Module = self._load_model_architecture()
        self.model.to(self.device)

        self.feature_extraction_layer_name = self.dgo_cfg.feature_extraction_layer_name
        if not self.feature_extraction_layer_name:
            # Default feature layer based on model type if not specified
            if isinstance(self.model, BaseCNN):
                self.feature_extraction_layer_name = "relu3"
            elif isinstance(self.model, ResNetForChars):  # Or ResNet base class
                self.feature_extraction_layer_name = "flatten_after_avgpool"
            else:
                logger.warning(
                    "Feature extraction layer not specified and could not be defaulted. Feature extraction might fail.")
                self.feature_extraction_layer_name = None  # Explicitly None

        if self.dgo_cfg.pretrained_model_path and os.path.exists(self.dgo_cfg.pretrained_model_path):
            self.load_weights(self.dgo_cfg.pretrained_model_path)
        else:
            logger.warning(
                f"No pretrained DGO model path provided or path invalid: {self.dgo_cfg.pretrained_model_path}. Model has random weights. Training is required.")
            # In a full system, an initial training step would be triggered here or by the TrainingEngine.

        self.model.eval()  # Default to evaluation mode

    def _load_model_architecture(self) -> nn.Module:
        """Loads the model architecture based on configuration."""
        model_class_or_factory = MODEL_REGISTRY.get(self.dgo_cfg.model_architecture)
        if not model_class_or_factory:
            raise ValueError(f"Unsupported DGO model architecture: {self.dgo_cfg.model_architecture}")

        logger.info(f"Loading DGO model architecture: {self.dgo_cfg.model_architecture}")

        # Some models might need more specific params from config
        # Example for ResNetVariant using its static factory method
        if self.dgo_cfg.model_architecture == "ResNetVariant":
            # The factory method ResNetForChars.resnet18_char_variant expects specific args
            return model_class_or_factory(
                num_classes=self.dgo_cfg.num_classes,
                input_channels=1 if self.data_cfg.grayscale_input else 3,
                # Pass any other specific params from dgo_cfg if needed by the factory
                initial_conv_stride=getattr(self.dgo_cfg, 'resnet_initial_conv_stride', 1),
                initial_pool_enabled=getattr(self.dgo_cfg, 'resnet_initial_pool_enabled', False),
                fc_features_out=getattr(self.dgo_cfg, 'resnet_fc_features_out',
                                        512 if isinstance(model_class_or_factory.__self__, ResNetForChars) else None)
                # Pass ResNet's fc.in_features as desired feature dim
            )
        elif self.dgo_cfg.model_architecture == "BaseCNN":
            return model_class_or_factory(
                num_classes=self.dgo_cfg.num_classes,
                input_channels=1 if self.data_cfg.grayscale_input else 3,
                img_size=self.data_cfg.target_image_size,
                fc_features=getattr(self.dgo_cfg, 'base_cnn_fc_features', 128),
                dropout_rate=getattr(self.dgo_cfg, 'base_cnn_dropout', 0.5)
            )
        else:  # Generic instantiation, assumes constructor matches
            return model_class_or_factory(
                num_classes=self.dgo_cfg.num_classes,
                input_channels=1 if self.data_cfg.grayscale_input else 3,
                # Add other common params if applicable
            )

    def load_weights(self, model_path: str, strict: bool = True):
        """Loads model weights from a file."""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle DataParallel-wrapped models if saved that way
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=strict)
            logger.info(f"DGO model weights loaded from {model_path}")
            self.model.eval()  # Ensure eval mode after loading
        except FileNotFoundError:
            logger.error(f"Model weights file not found: {model_path}")
            raise
        except RuntimeError as e:  # Mismatched keys etc.
            logger.error(f"Error loading model weights from {model_path}: {e}")
            if not strict:
                logger.warning("Attempted to load with strict=False. Some weights might be missing/unexpected.")
            else:
                raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading weights: {e}")
            raise

    def save_weights(self, model_path: str):
        """Saves model weights to a file."""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # If model is wrapped (e.g. DataParallel), save its module's state_dict
            if isinstance(self.model, nn.DataParallel):
                torch.save(self.model.module.state_dict(), model_path)
            else:
                torch.save(self.model.state_dict(), model_path)
            logger.info(f"DGO model weights saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving DGO model weights to {model_path}: {e}")
            raise

    @contextmanager
    def _training_mode_if_needed(self, for_mc_dropout: bool = False):
        """Context manager to temporarily set model to train() for MC Dropout, then restore."""
        original_mode_is_training = self.model.training
        if for_mc_dropout and self.dgo_cfg.uncertainty_method == "mc_dropout":
            if not hasattr(self.model, 'dropout') and not any(isinstance(m, nn.Dropout) for m in self.model.modules()):
                logger.warning(
                    "MC Dropout requested, but no Dropout layers found in the model. Uncertainty might be unreliable.")
            self.model.train()  # Enable dropout layers
        else:  # Keep original mode (usually eval)
            pass
        try:
            yield
        finally:
            if original_mode_is_training:
                self.model.train()
            else:
                self.model.eval()

    def predict_raw_logits(self, image_input: Union[CvImage, PILImage, str, torch.Tensor],
                           extract_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Internal method for getting raw logits and optionally features.
        Input can be preprocessed tensor or raw image.
        """
        if not isinstance(image_input, torch.Tensor):
            # Preprocess raw image (path, PIL, CvImage) to tensor
            # This uses the DGO's specific preprocessing pipeline
            input_tensor = preprocess_image_for_dgo(
                image_input,
                target_size=self.data_cfg.target_image_size,
                grayscale=self.data_cfg.grayscale_input,
                device=self.device
            )
        else:  # Assume input_tensor is already preprocessed and on the correct device
            input_tensor = image_input.to(self.device)
            if input_tensor.ndim == 3:  # (C,H,W) -> (1,C,H,W)
                input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():  # Standard prediction doesn't need gradients here
            # The model's forward method should handle feature extraction if layer name is passed
            feature_layers_to_extract = [
                self.feature_extraction_layer_name] if extract_features and self.feature_extraction_layer_name else None

            if hasattr(self.model,
                       'forward') and 'extract_features_layer_names' in self.model.forward.__code__.co_varnames:
                logits = self.model(input_tensor, extract_features_layer_names=feature_layers_to_extract)
            else:  # Model forward doesn't support explicit feature layer names
                logits = self.model(input_tensor)
                # Feature extraction would need a separate call or hook based features.
                # This handler assumes models used have the custom forward or get_feature_vector.
                if extract_features and not feature_layers_to_extract:
                    logger.warning(
                        "Model's forward does not support feature extraction arg, or no feature layer specified.")

            features_tensor: Optional[torch.Tensor] = None
            if extract_features and self.feature_extraction_layer_name:
                if hasattr(self.model, 'get_last_extracted_features'):  # For models like BaseCNN/ResNetForChars
                    extracted_map = self.model.get_last_extracted_features()
                    features_tensor = extracted_map.get(self.feature_extraction_layer_name)
                    if features_tensor is not None:
                        features_tensor = torch.flatten(features_tensor, 1)  # Ensure flat vector
                elif hasattr(self.model, 'get_feature_vector'):  # Alternative interface
                    features_tensor = self.model.get_feature_vector(input_tensor,
                                                                    layer_name=self.feature_extraction_layer_name)
                else:
                    logger.warning(
                        f"Model {type(self.model)} does not have get_last_extracted_features or get_feature_vector methods. Cannot extract features this way.")

        return logits, features_tensor

    def predict(self, image_input: Union[CvImage, PILImage, str, torch.Tensor]) -> DGOOutput:
        """
        Performs prediction on a single image.
        Returns: (predicted_class_idx, confidence, all_class_probabilities, feature_vector, uncertainty_metric)
        """
        uncertainty_val: Optional[DGOUncertainty] = None

        with self._training_mode_if_needed(for_mc_dropout=True):  # Enable train() for MC Dropout
            if self.dgo_cfg.uncertainty_method == "mc_dropout" and self.dgo_cfg.mc_dropout_samples > 1:
                # Monte Carlo Dropout for uncertainty
                all_logits_mc = []
                all_features_mc = []  # If features are also averaged over MC samples

                for _ in range(self.dgo_cfg.mc_dropout_samples):
                    # Need to ensure model is in train mode for dropout to be active
                    # self._training_mode_if_needed handles this
                    logits_i, features_i = self.predict_raw_logits(image_input, extract_features=True)
                    all_logits_mc.append(logits_i)
                    if features_i is not None:
                        all_features_mc.append(features_i)

                stacked_logits = torch.stack(all_logits_mc)  # (N_samples, B, Num_classes), B=1 here

                # Mean logits and then softmax, or softmax then mean probabilities
                # Softmax then mean is more common for classification probability
                stacked_probs = F.softmax(stacked_logits, dim=-1)  # Softmax over class dimension
                mean_probs = torch.mean(stacked_probs, dim=0)  # Average over N_samples -> (B, Num_classes)

                # Uncertainty: Predictive Entropy or Variance of probabilities
                # Predictive Entropy: -sum(p_mean * log(p_mean))
                predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9),
                                                dim=-1).item()  # For batch_size 1
                # Or Variance of softmax outputs across samples
                # variance_of_probs = torch.var(stacked_probs, dim=0).sum(dim=-1).item() # Sum variance over classes
                uncertainty_val = predictive_entropy  # Choose one metric

                final_probs = mean_probs.squeeze(0)  # (Num_classes)

                # Average features if extracted during MC
                features_np: Optional[FeatureVector] = None
                if all_features_mc and all_features_mc[0] is not None:
                    mean_features = torch.mean(torch.stack(all_features_mc), dim=0)
                    features_np = mean_features.cpu().numpy().squeeze()

            else:  # Standard single-pass prediction
                logits, features_tensor = self.predict_raw_logits(image_input, extract_features=True)
                final_probs_torch = F.softmax(logits, dim=1).squeeze(0)  # (Num_classes)
                final_probs = final_probs_torch.cpu().numpy()

                features_np: Optional[FeatureVector] = None
                if features_tensor is not None:
                    features_np = features_tensor.cpu().numpy().squeeze()

                if self.dgo_cfg.uncertainty_method != "none" and self.dgo_cfg.uncertainty_method != "mc_dropout":
                    logger.warning(
                        f"Uncertainty method '{self.dgo_cfg.uncertainty_method}' not fully implemented for single pass.")
                # For single pass, can use max_softmax_prob as inverse uncertainty, or entropy of single prediction
                # uncertainty_val = -torch.sum(final_probs_torch * torch.log(final_probs_torch + 1e-9)).item()

        # Convert to numpy for output consistency
        probabilities_np: DGOScores = final_probs if isinstance(final_probs, np.ndarray) else final_probs.cpu().numpy()

        confidence, pred_idx = np.max(probabilities_np), np.argmax(probabilities_np)

        return pred_idx, float(confidence), probabilities_np, features_np, uncertainty_val

    def get_gradient_wrt_input(self,
                               image_input: Union[CvImage, PILImage, str, torch.Tensor],
                               target_class_idx: Optional[int] = None,
                               loss_type: str = "logit_target_class",  # "prob_target_class", "cross_entropy_vs_target"
                               misguide_to_class_idx: Optional[int] = None
                               ) -> np.ndarray:
        """
        Calculates the gradient of a chosen loss/output w.r.t. the input image.
        This is used for crafting adversarial perturbations.
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
        self.model.zero_grad()  # Clear any existing gradients in the model

        # Ensure model is in eval mode for consistent gradient calculation unless specified
        # (e.g. if dropout should be active for some gradient-based uncertainty)
        # For standard gradients, eval mode is typical.
        # _training_mode_if_needed not used here as we usually want deterministic gradients.
        original_training_state = self.model.training
        self.model.eval()  # Set to eval for gradient calculation

        # Forward pass
        logits = self.model(input_tensor)  # No feature extraction needed for gradients usually

        loss: Optional[torch.Tensor] = None
        if misguide_to_class_idx is not None:
            # Objective: Maximize probability of misguide_to_class_idx
            # Or minimize probability of true_target_class_idx (if provided alongside misguide)
            if loss_type == "logit_target_class":
                loss = -logits[0, misguide_to_class_idx]  # Maximize this logit (minimize negative)
            elif loss_type == "prob_target_class":
                probs = F.softmax(logits, dim=1)
                loss = -probs[0, misguide_to_class_idx]
            else:  # Default to logit for misguide
                loss = -logits[0, misguide_to_class_idx]
        elif target_class_idx is not None:
            # Objective: Maximize probability of target_class_idx (if reinforcing)
            # Or if we want to find perturbations that *reduce* confidence in target_class_idx (Fast Gradient Sign Method direction)
            # For FGSM, we take gradient of loss w.r.t input. If loss is CrossEntropy(output, target_label),
            # then input_adv = input + eps * sign(grad).
            # If we want to make it *less* like target_class_idx using its own logit:
            if loss_type == "logit_target_class":
                loss = logits[
                    0, target_class_idx]  # To find perturbation that *decreases* this, gradient ascent on this loss.
                # Or, -logits[0, target_idx] and gradient descent.
            elif loss_type == "prob_target_class":
                probs = F.softmax(logits, dim=1)
                loss = probs[0, target_class_idx]
            elif loss_type == "cross_entropy_vs_target":
                # target_tensor = torch.tensor([target_class_idx], device=self.device)
                # loss = F.cross_entropy(logits, target_tensor) # This is for training.
                # For gradient direction to *increase* CE (make it less like target):
                target_tensor = torch.tensor([target_class_idx], device=self.device)
                loss = F.nll_loss(F.log_softmax(logits, dim=1), target_tensor)  # Standard loss for classification.
                # Gradient of this loss w.r.t. input tells how to change input to reduce this loss (i.e., make it more like target).
                # To make it *less* like target, you could use -loss, or use the gradient as is for FGSM-like attacks.
            else:
                loss = logits[0, target_class_idx]  # Default
        else:
            # No target specified, perhaps gradient of sum of logits or max logit?
            # This case needs specific definition. For now, raise error.
            self.model.train(original_training_state)  # Restore mode
            raise ValueError(
                "Either target_class_idx or misguide_to_class_idx must be provided for gradient calculation.")

        if loss is None:  # Should not happen if logic above is correct
            self.model.train(original_training_state)
            raise ValueError("Loss for gradient calculation was not defined.")

        loss.backward()  # Compute gradients

        gradient = input_tensor.grad.data.cpu().numpy().squeeze()

        self.model.train(original_training_state)  # Restore original model mode (train/eval)

        return gradient  # (H, W) or (C, H, W) numpy array

    def get_feature_dimensionality(self) -> Optional[int]:
        """
        Tries to infer the dimensionality of the feature vector extracted by the DGO.
        This is important for initializing the FeatureHasher.
        """
        if not self.feature_extraction_layer_name:
            logger.warning("Cannot determine feature dimensionality: feature_extraction_layer_name not set.")
            return None

        # Create a dummy input tensor
        # Use data_config for image size and channels
        channels = 1 if self.data_cfg.grayscale_input else 3
        h, w = self.data_cfg.target_image_size
        dummy_input = torch.randn(1, channels, h, w).to(self.device)

        self.model.eval()  # Ensure eval mode
        with torch.no_grad():
            features_tensor: Optional[torch.Tensor] = None
            if hasattr(self.model,
                       'forward') and 'extract_features_layer_names' in self.model.forward.__code__.co_varnames:
                _ = self.model(dummy_input, extract_features_layer_names=[self.feature_extraction_layer_name])
                if hasattr(self.model, 'get_last_extracted_features'):
                    extracted_map = self.model.get_last_extracted_features()
                    features_tensor = extracted_map.get(self.feature_extraction_layer_name)
            elif hasattr(self.model, 'get_feature_vector'):
                features_tensor = self.model.get_feature_vector(dummy_input,
                                                                layer_name=self.feature_extraction_layer_name)

            if features_tensor is not None:
                # Features might be (B, C, Hf, Wf) or (B, Dim)
                # Flatten to get the final dimension
                return torch.flatten(features_tensor, 1).shape[1]
            else:
                logger.warning(
                    f"Could not extract features using layer '{self.feature_extraction_layer_name}' to determine dimensionality.")
                return None


if __name__ == "__main__":
    # --- Test DGOModelHandler ---
    from ..config import SystemConfig  # Full system config

    # Create a temporary DGO config for BaseCNN
    temp_sys_cfg_data = {
        "project_name": "DGOHandlerTest",
        "data_management": {"target_image_size": (28, 28), "grayscale_input": True},
        "dgo_oracle": {
            "model_architecture": "BaseCNN",  # Test with BaseCNN first
            "num_classes": 10,
            "pretrained_model_path": None,  # No pretrained for this test
            "feature_extraction_layer_name": "relu3",  # For BaseCNN
            "uncertainty_method": "mc_dropout",  # Test MC Dropout
            "mc_dropout_samples": 5,
            "base_cnn_fc_features": 64,  # Make it smaller for test
            "base_cnn_dropout": 0.25  # Ensure dropout is present
        },
        "logging": {"level": "DEBUG"}
    }
    # Override global config for this test script
    from ..config import _config_instance

    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data)

    test_dgo_cfg = get_config().dgo_oracle
    test_data_cfg = get_config().data_management
    test_device = get_config().get_actual_device()

    logger = setup_logging()  # Re-init logger with new config

    print(f"--- Testing DGOModelHandler with {test_dgo_cfg.model_architecture} ---")
    handler = DGOModelHandler(test_dgo_cfg, test_data_cfg, device=test_device)

    # Test feature dimensionality inference
    feat_dim = handler.get_feature_dimensionality()
    print(f"Inferred feature dimensionality: {feat_dim}")
    assert feat_dim == getattr(test_dgo_cfg, 'base_cnn_fc_features', 128)  # Check against BaseCNN's fc_features

    # Create a dummy image (CvImage uint8)
    dummy_cv_image = np.random.randint(0, 256, test_data_cfg.target_image_size, dtype=np.uint8)

    # Test prediction
    print("\nTesting predict()...")
    pred_idx, conf, probs, features, uncertainty = handler.predict(dummy_cv_image)
    print(f"  Pred Idx: {pred_idx}, Confidence: {conf:.4f}")
    print(f"  Probabilities shape: {probs.shape}")
    assert probs.shape == (test_dgo_cfg.num_classes,)
    if features is not None:
        print(f"  Features shape: {features.shape}")
        assert features.shape == (feat_dim,)  # Check inferred dim
    else:
        print("  Features: None (this might be an issue if expected)")
        assert handler.feature_extraction_layer_name is None  # or a problem

    print(f"  Uncertainty ({test_dgo_cfg.uncertainty_method}): {uncertainty}")
    if test_dgo_cfg.uncertainty_method == "mc_dropout":
        assert uncertainty is not None

    # Test gradient calculation
    print("\nTesting get_gradient_wrt_input()...")
    target_cls_for_grad = 3
    gradient = handler.get_gradient_wrt_input(dummy_cv_image, target_class_idx=target_cls_for_grad,
                                              loss_type="logit_target_class")
    print(f"  Gradient shape: {gradient.shape}")  # Expected (H, W) for grayscale
    expected_grad_shape = test_data_cfg.target_image_size if test_data_cfg.grayscale_input else (
    3, test_data_cfg.target_image_size[0], test_data_cfg.target_image_size[1])
    # Gradient is (C,H,W) from tensor, squeezed if C=1 for grayscale.
    # preprocess_image_for_dgo makes it (1,C,H,W), grad is (C,H,W) then squeezed if C=1
    # So for grayscale, (H,W). For color, (3,H,W).
    # The current squeeze in get_gradient_wrt_input will make grayscale (H,W).
    if test_data_cfg.grayscale_input:
        assert gradient.shape == test_data_cfg.target_image_size
    else:  # Color
        assert gradient.shape == (3, test_data_cfg.target_image_size[0], test_data_cfg.target_image_size[1])

    # --- Test with ResNetVariant ---
    if "ResNetVariant" in MODEL_REGISTRY:  # Check if it's available
        temp_sys_cfg_data_resnet = temp_sys_cfg_data.copy()  # Start from base
        temp_sys_cfg_data_resnet["dgo_oracle"]["model_architecture"] = "ResNetVariant"
        temp_sys_cfg_data_resnet["dgo_oracle"]["feature_extraction_layer_name"] = "flatten_after_avgpool"
        # ResNet specific params (defaults are in ResNetForChars.resnet18_char_variant)
        temp_sys_cfg_data_resnet["dgo_oracle"]["resnet_initial_conv_stride"] = 1
        temp_sys_cfg_data_resnet["dgo_oracle"]["resnet_initial_pool_enabled"] = False
        temp_sys_cfg_data_resnet["dgo_oracle"]["resnet_fc_features_out"] = 512  # ResNet18 output before FC

        _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_resnet)
        test_dgo_cfg_res = get_config().dgo_oracle
        logger = setup_logging()  # Re-init for new config context if needed

        print(f"\n--- Testing DGOModelHandler with {test_dgo_cfg_res.model_architecture} ---")
        handler_res = DGOModelHandler(test_dgo_cfg_res, test_data_cfg, device=test_device)

        feat_dim_res = handler_res.get_feature_dimensionality()
        print(f"Inferred feature dimensionality (ResNet): {feat_dim_res}")
        assert feat_dim_res == test_dgo_cfg_res.resnet_fc_features_out

        pred_idx_r, conf_r, _, features_r, uncertainty_r = handler_res.predict(dummy_cv_image)
        print(f"  ResNet - Pred Idx: {pred_idx_r}, Confidence: {conf_r:.4f}, Uncertainty: {uncertainty_r}")
        if features_r is not None:
            assert features_r.shape == (feat_dim_res,)
        else:
            assert handler_res.feature_extraction_layer_name is None

    print("\nDGOModelHandler tests completed.")