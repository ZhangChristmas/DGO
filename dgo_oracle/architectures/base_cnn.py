# ultimate_morph_generator/dgo_oracle/architectures/base_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BaseCNN(nn.Module):
    """
    A simple Convolutional Neural Network, adaptable for character recognition.
    Similar to LeNet, but with configurable parameters.
    """

    def __init__(self,
                 num_classes: int = 10,
                 input_channels: int = 1,
                 img_size: Tuple[int, int] = (28, 28),  # H, W
                 fc_features: int = 128,
                 dropout_rate: float = 0.5):
        super(BaseCNN, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_height, self.img_width = img_size
        self.fc_features = fc_features

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)  # Keep size
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves size

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # Keep size
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves size again

        # Calculate flattened size after convolutions and pooling
        # After pool1: img_size / 2
        # After pool2: img_size / 4
        self.flattened_dim = 64 * (self.img_height // 4) * (self.img_width // 4)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_dim, fc_features)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_features, num_classes)  # Output logits

        # For feature extraction (can be defined based on layer name)
        self.feature_hooks = {}  # To store hooks if needed dynamically
        self._features_cache = {}  # To store features from specified layers

    def forward(self, x: torch.Tensor,
                extract_features_layer_names: Optional[list[str]] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor (B, C, H, W).
            extract_features_layer_names: List of layer names from which to extract features.
                                          E.g., ["relu3", "pool2"].
        Returns:
            Logits tensor (B, num_classes).
        """
        # Clear previous features cache if new extraction is requested
        if extract_features_layer_names:
            self._features_cache = {}

        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        if extract_features_layer_names and "pool1" in extract_features_layer_names:
            self._features_cache["pool1"] = x.detach()
        if extract_features_layer_names and "conv1_relu" in extract_features_layer_names:  # Example of combined name
            self._features_cache["conv1_relu"] = x.detach()

        # Block 2
        x = self.conv2(x)
        if extract_features_layer_names and "conv2" in extract_features_layer_names:
            self._features_cache["conv2"] = x.detach()
        x = self.relu2(x)
        if extract_features_layer_names and "relu2" in extract_features_layer_names:
            self._features_cache["relu2"] = x.detach()
        x = self.pool2(x)
        if extract_features_layer_names and "pool2" in extract_features_layer_names:
            self._features_cache["pool2"] = x.detach()

        # Flatten
        x = torch.flatten(x, 1)
        if x.shape[1] != self.flattened_dim:  # Dynamic check, should match calculation
            # This can happen if img_size is not perfectly divisible by 4
            # A more robust way is to use nn.AdaptiveAvgPool2d before fc1 or compute shape dynamically
            # For now, assume it matches or adjust self.flattened_dim in __init__
            # For robustness: self.fc1 = nn.Linear(x.shape[1], self.fc_features) # if defined in forward
            raise ValueError(
                f"Flattened dimension mismatch: expected {self.flattened_dim}, got {x.shape[1]}. Check img_size and conv/pool layers.")

        # FC Block
        x = self.fc1(x)
        if extract_features_layer_names and "fc1" in extract_features_layer_names:
            self._features_cache["fc1"] = x.detach()
        x = self.relu3(x)
        if extract_features_layer_names and "relu3" in extract_features_layer_names:  # This is a common feature layer
            self._features_cache["relu3"] = x.detach()

        x = self.dropout(x)
        logits = self.fc2(x)  # Output logits

        return logits

    def get_last_extracted_features(self) -> dict[str, torch.Tensor]:
        """Returns the cached features from the last forward pass with extraction."""
        return self._features_cache

    def get_feature_vector(self, x: torch.Tensor, layer_name: str = "relu3") -> Optional[torch.Tensor]:
        """
        Convenience method to get features from a specific layer after a forward pass.
        Returns a flattened feature vector.
        """
        _ = self.forward(x, extract_features_layer_names=[layer_name])
        features = self._features_cache.get(layer_name)
        if features is not None:
            return torch.flatten(features, 1)
        return None

    @staticmethod
    def from_config(dgo_cfg, data_cfg):  # Helper to instantiate from config
        return BaseCNN(
            num_classes=dgo_cfg.num_classes,
            input_channels=1 if data_cfg.grayscale_input else 3,
            img_size=data_cfg.target_image_size,
            fc_features=getattr(dgo_cfg, 'base_cnn_fc_features', 128),  # Example specific param
            dropout_rate=getattr(dgo_cfg, 'base_cnn_dropout', 0.5)
        )


if __name__ == '__main__':
    # Test the BaseCNN model
    num_classes = 10
    input_channels = 1
    img_h, img_w = 28, 28

    model = BaseCNN(num_classes=num_classes, input_channels=input_channels, img_size=(img_h, img_w))
    print("BaseCNN Model Structure:")
    print(model)

    # Test forward pass
    dummy_batch_size = 4
    dummy_input = torch.randn(dummy_batch_size, input_channels, img_h, img_w)

    print(f"\nInput tensor shape: {dummy_input.shape}")

    # Forward pass without feature extraction
    logits = model(dummy_input)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (dummy_batch_size, num_classes)

    # Forward pass with feature extraction
    feature_layer_to_extract = "relu3"  # Commonly used feature layer
    logits_with_feat = model(dummy_input, extract_features_layer_names=[feature_layer_to_extract, "pool2"])
    extracted_features_map = model.get_last_extracted_features()

    print(f"\nExtracted features for layers: {list(extracted_features_map.keys())}")
    if feature_layer_to_extract in extracted_features_map:
        relu3_features = extracted_features_map[feature_layer_to_extract]
        print(f"Features from '{feature_layer_to_extract}' shape: {relu3_features.shape}")
        # For relu3, expected shape: (batch_size, fc_features)
        assert relu3_features.shape == (dummy_batch_size, model.fc_features)

    if "pool2" in extracted_features_map:
        pool2_features = extracted_features_map["pool2"]
        print(f"Features from 'pool2' shape: {pool2_features.shape}")
        # For pool2, expected shape: (batch_size, 64, img_h // 4, img_w // 4)
        assert pool2_features.shape == (dummy_batch_size, 64, img_h // 4, img_w // 4)

    # Test convenience get_feature_vector
    features_vec = model.get_feature_vector(dummy_input, layer_name=feature_layer_to_extract)
    if features_vec is not None:
        print(f"\nFeature vector from '{feature_layer_to_extract}' (flattened) shape: {features_vec.shape}")
        assert features_vec.shape == (dummy_batch_size, model.fc_features)

    print("\nBaseCNN tests completed.")