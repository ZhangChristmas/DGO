# ultimate_morph_generator/dgo_oracle/architectures/resnet_variant.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet  # type: ignore
from typing import Optional, Tuple, List, Type, Union


# We can use torchvision's ResNet and adapt its first conv layer and FC layer.
# Or define a smaller custom ResNet. Let's try adapting torchvision's.

class ResNetForChars(ResNet):
    """
    Adapts a torchvision ResNet (e.g., ResNet18) for character recognition tasks
    by modifying the initial convolution for small input images and the final FC layer.
    """

    def __init__(self, block: Type[Union[BasicBlock]], layers: List[int],
                 num_classes: int = 10, input_channels: int = 1,
                 initial_conv_stride: int = 1, initial_pool_enabled: bool = False,
                 fc_features_out: Optional[int] = None):  # fc_features_out for DGO feature extraction
        super(ResNetForChars, self).__init__(block, layers, num_classes=num_classes)

        # Modify the first convolutional layer for small, potentially grayscale images
        # Original ResNet conv1: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # For 28x28 or 32x32 images, stride=2 and kernel=7 is too aggressive.
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                               stride=initial_conv_stride, padding=1, bias=False)

        # Original ResNet maxpool: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # This might also be too aggressive for small images. We can disable or change it.
        if not initial_pool_enabled:
            self.maxpool = nn.Identity()  # Effectively removes the initial max pooling

        # Modify the final fully connected layer if num_classes changed
        # The number of input features to self.fc depends on the ResNet variant (e.g., 512 for ResNet18/34)
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

        # For feature extraction (can be adapted from BaseCNN's method)
        self._features_cache = {}
        self.fc_features_out = fc_features_out if fc_features_out is not None else num_ftrs

        # If we want to extract features before the final fc layer,
        # we can replace self.fc with an Identity and add a new classification head.
        # Or, more simply, hook into the layer before self.fc (usually avgpool or flatten).
        # Torchvision ResNet has `self.avgpool` (AdaptiveAvgPool2d) before `self.fc`.

    def _forward_impl(self, x: torch.Tensor,
                      extract_features_layer_names: Optional[list[str]] = None) -> torch.Tensor:
        # See note [TorchScript super()]
        if extract_features_layer_names:
            self._features_cache = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if extract_features_layer_names and "stem_relu" in extract_features_layer_names:
            self._features_cache["stem_relu"] = x.detach()

        x = self.maxpool(x)
        if extract_features_layer_names and "stem_maxpool" in extract_features_layer_names:
            self._features_cache["stem_maxpool"] = x.detach()

        x = self.layer1(x)
        if extract_features_layer_names and "layer1" in extract_features_layer_names:
            self._features_cache["layer1"] = x.detach()
        x = self.layer2(x)
        if extract_features_layer_names and "layer2" in extract_features_layer_names:
            self._features_cache["layer2"] = x.detach()
        x = self.layer3(x)
        if extract_features_layer_names and "layer3" in extract_features_layer_names:
            self._features_cache["layer3"] = x.detach()
        x = self.layer4(x)
        if extract_features_layer_names and "layer4" in extract_features_layer_names:
            self._features_cache["layer4"] = x.detach()

        x = self.avgpool(x)  # AdaptiveAvgPool2d((1,1))
        if extract_features_layer_names and "avgpool" in extract_features_layer_names:
            self._features_cache["avgpool"] = x.detach()  # Shape (B, C, 1, 1)

        x = torch.flatten(x, 1)  # Flatten for FC
        if extract_features_layer_names and "flatten_after_avgpool" in extract_features_layer_names:
            self._features_cache["flatten_after_avgpool"] = x.detach()  # Shape (B, C) - this is a good feature vector

        logits = self.fc(x)
        return logits

    def forward(self, x: torch.Tensor,
                extract_features_layer_names: Optional[list[str]] = None) -> torch.Tensor:
        return self._forward_impl(x, extract_features_layer_names)

    def get_last_extracted_features(self) -> dict[str, torch.Tensor]:
        return self._features_cache

    def get_feature_vector(self, x: torch.Tensor, layer_name: str = "flatten_after_avgpool") -> Optional[torch.Tensor]:
        """ Default feature extraction is after avgpool and flatten. """
        _ = self.forward(x, extract_features_layer_names=[layer_name])
        features = self._features_cache.get(layer_name)
        if features is not None:
            # Already flattened if layer_name is "flatten_after_avgpool"
            # If layer_name was "avgpool", it would be (B,C,1,1), needs flatten
            if layer_name == "avgpool" and features.ndim == 4:
                return torch.flatten(features, 1)
            return features
        return None

    @staticmethod
    def resnet18_char_variant(num_classes: int, input_channels: int, **kwargs) -> "ResNetForChars":
        # fc_features_out is used by DGOHandler, ResNet itself doesn't use it directly.
        # For ResNet18, self.fc.in_features is 512.
        fc_out = kwargs.pop('fc_features_out', 512)
        return ResNetForChars(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                              input_channels=input_channels, fc_features_out=fc_out, **kwargs)

    @staticmethod
    def from_config(dgo_cfg, data_cfg):  # Helper to instantiate from config
        if dgo_cfg.model_architecture == "ResNetVariant":  # Check specific model name in future if more variants
            # Example of specific params from dgo_cfg for this model variant
            initial_conv_stride = getattr(dgo_cfg, 'resnet_initial_conv_stride', 1)
            initial_pool_enabled = getattr(dgo_cfg, 'resnet_initial_pool_enabled', False)
            fc_features_out_val = getattr(dgo_cfg, 'resnet_fc_features_out', None)  # Could be 512 for ResNet18

            return ResNetForChars.resnet18_char_variant(
                num_classes=dgo_cfg.num_classes,
                input_channels=1 if data_cfg.grayscale_input else 3,
                initial_conv_stride=initial_conv_stride,
                initial_pool_enabled=initial_pool_enabled,
                fc_features_out=fc_features_out_val
            )
        else:
            raise ValueError(
                f"Config requests ResNetVariant, but from_config called for other type or sub-type not specified.")


if __name__ == '__main__':
    num_classes = 10
    input_channels = 1  # Grayscale
    img_h, img_w = 32, 32  # Example size, ResNet can handle it

    # Create ResNet18 variant
    model = ResNetForChars.resnet18_char_variant(num_classes=num_classes, input_channels=input_channels,
                                                 initial_conv_stride=1, initial_pool_enabled=False)
    print("ResNet18 Char Variant Model Structure (부분적으로):")
    # print(model) # Full ResNet print is very long
    print(f"  conv1: {model.conv1}")
    print(f"  maxpool: {model.maxpool}")
    print(f"  fc: {model.fc}")
    print(f"  fc_features_out attribute: {model.fc_features_out}")

    dummy_batch_size = 2
    dummy_input = torch.randn(dummy_batch_size, input_channels, img_h, img_w)
    print(f"\nInput tensor shape: {dummy_input.shape}")

    # Forward pass
    logits = model(dummy_input)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (dummy_batch_size, num_classes)

    # Feature extraction test
    feature_layer = "flatten_after_avgpool"  # Default good feature layer
    logits_feat = model(dummy_input, extract_features_layer_names=[feature_layer, "layer2"])
    extracted_map = model.get_last_extracted_features()

    print(f"\nExtracted features for layers: {list(extracted_map.keys())}")
    if feature_layer in extracted_map:
        features = extracted_map[feature_layer]
        print(f"Features from '{feature_layer}' shape: {features.shape}")
        # For ResNet18, this is 512 features
        assert features.shape[1] == 512 if model.fc.in_features == 512 else True

    if "layer2" in extracted_map:
        layer2_features = extracted_map["layer2"]
        print(f"Features from 'layer2' shape: {layer2_features.shape}")
        # For ResNet18, layer2 output has 128 channels. Size depends on strides.
        # With conv1 stride 1, no initial pool, layer1/2 strides of 1 (BasicBlock default):
        # H/1 (conv1_stride) /1 (maxpool_ident) /1 (layer1_stride) /2 (layer2_stride) = H/2
        # Shape: (B, 128, img_h // 2, img_w // 2) if layer2 has stride 2 in its first block
        # The actual output shape depends on the ResNet block configuration.
        # Example ResNet18: layer1 out: 64 x H/1 x W/1 (if no pooling/stride1 in stem)
        # layer2 out: 128 x H/2 x W/2 (first block of layer2 usually has stride 2)
        # So for 32x32 input: (B, 128, 16, 16)
        assert layer2_features.shape[1] == 128
        # assert layer2_features.shape[2:] == (img_h // 2, img_w // 2) # This assertion can be tricky due to ResNet structure

    # Test convenience get_feature_vector
    features_vec = model.get_feature_vector(dummy_input, layer_name=feature_layer)
    if features_vec is not None:
        print(f"\nFeature vector from '{feature_layer}' (flattened) shape: {features_vec.shape}")
        assert features_vec.shape[1] == model.fc_features_out

    print("\nResNetForChars (ResNet18 Variant) tests completed.")