# ultimate_morph_generator/dgo_oracle/architectures/vit_small.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# Helper for Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (28, 28), patch_size: int = 4,
                 in_channels: int = 1, embed_dim: int = 192):  # embed_dim example
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Linear projection for patches
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # (B, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (B, embed_dim, n_patches_h * n_patches_w)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim) -> Standard ViT sequence format
        return x


# Basic Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)  # For skip connection part if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention part
        res_x = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)  # Query, Key, Value are the same
        x = self.dropout(x)  # dropout after attention, common practice
        x = x + res_x  # Skip connection

        # MLP part
        res_x = x
        x = self.norm2(x)
        x = self.mlp(x)
        # x = self.dropout(x) # dropout after MLP (already in MLP last layer)
        x = x + res_x  # Skip connection
        return x


class ViTSmallForChars(nn.Module):
    """
    A small Vision Transformer (ViT) adapted for character recognition.
    This is a conceptual implementation. For production, use established libraries like `timm`.
    """

    def __init__(self,
                 img_size: Tuple[int, int] = (28, 28),
                 patch_size: int = 4,  # e.g., 28x28 image, 4x4 patches -> (28/4)*(28/4) = 7*7 = 49 patches
                 in_channels: int = 1,
                 num_classes: int = 10,
                 embed_dim: int = 192,  # Dimension of token embeddings (and thus, model width)
                 depth: int = 6,  # Number of Transformer encoder blocks
                 num_heads: int = 6,  # Number of attention heads
                 mlp_ratio: float = 3.0,  # Ratio for MLP hidden layer size
                 dropout: float = 0.1,
                 use_class_token: bool = True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_class_token = use_class_token

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Class token and positional embeddings
        if self.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Learnable [CLS] token
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for CLS token
        else:  # Use mean pooling of patch tokens instead of CLS token
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.pos_dropout = nn.Dropout(p=dropout)

        # Transformer Encoder stack
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)  # Final normalization

        # Classification head
        # If using CLS token, head is on self.cls_token.
        # If using mean pooling, head is on the mean of patch tokens.
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights (important for ViTs)
        self._initialize_weights()

        # For feature extraction
        self._features_cache: Dict[str, torch.Tensor] = {}

    def _initialize_weights(self):
        # Truncated normal initialization for positional embeddings and CLS token
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=.02)

        # Initialize linear layers and LayerNorms
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):  # Per-module init
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):  # For patch embedding projection
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Or trunc_normal_
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                extract_features_layer_names: Optional[List[str]] = None) -> torch.Tensor:
        B = x.shape[0]  # Batch size
        if extract_features_layer_names: self._features_cache = {}

        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Pass through Transformer encoder blocks
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if extract_features_layer_names and f"encoder_block_{i}" in extract_features_layer_names:
                self._features_cache[f"encoder_block_{i}"] = x.detach()

        x = self.norm(x)  # Final normalization on the sequence
        if extract_features_layer_names and "final_norm_seq" in extract_features_layer_names:
            self._features_cache["final_norm_seq"] = x.detach()  # Full sequence output

        # Classification
        if self.use_class_token and self.cls_token is not None:
            # Use the CLS token's output embedding for classification
            cls_token_final_embedding = x[:, 0]  # (B, embed_dim)
            if extract_features_layer_names and "cls_token_embedding" in extract_features_layer_names:
                self._features_cache["cls_token_embedding"] = cls_token_final_embedding.detach()
            logits = self.head(cls_token_final_embedding)
        else:
            # Use mean pooling of patch tokens (excluding CLS token if it were present but not used for head)
            patch_tokens_final_embedding = x.mean(dim=1)  # (B, embed_dim) - Average over sequence dim
            if extract_features_layer_names and "mean_patch_embedding" in extract_features_layer_names:
                self._features_cache["mean_patch_embedding"] = patch_tokens_final_embedding.detach()
            logits = self.head(patch_tokens_final_embedding)

        return logits

    def get_last_extracted_features(self) -> dict[str, torch.Tensor]:
        return self._features_cache

    def get_feature_vector(self, x: torch.Tensor, layer_name: str = "cls_token_embedding") -> Optional[torch.Tensor]:
        if not self.use_class_token and layer_name == "cls_token_embedding":
            layer_name = "mean_patch_embedding"  # Default if no CLS token
            logger.debug(f"ViT: CLS token not used, defaulting feature extraction to '{layer_name}'.")

        _ = self.forward(x, extract_features_layer_names=[layer_name])
        features = self._features_cache.get(layer_name)
        if features is not None:
            # Already (B, Dim) if 'cls_token_embedding' or 'mean_patch_embedding'
            return features
        return None

    @staticmethod
    def from_config(dgo_cfg, data_cfg):
        # Example of how to get ViT specific params from config
        return ViTSmallForChars(
            img_size=data_cfg.target_image_size,
            patch_size=getattr(dgo_cfg, 'vit_patch_size', 4),
            in_channels=1 if data_cfg.grayscale_input else 3,
            num_classes=dgo_cfg.num_classes,
            embed_dim=getattr(dgo_cfg, 'vit_embed_dim', 192),
            depth=getattr(dgo_cfg, 'vit_depth', 6),
            num_heads=getattr(dgo_cfg, 'vit_num_heads', 6),
            mlp_ratio=getattr(dgo_cfg, 'vit_mlp_ratio', 3.0),
            dropout=getattr(dgo_cfg, 'vit_dropout', 0.1),
            use_class_token=getattr(dgo_cfg, 'vit_use_class_token', True)
        )


if __name__ == '__main__':
    # Test ViTSmallForChars
    test_img_size = (32, 32)  # Needs to be divisible by patch_size
    test_patch_size = 4
    test_in_channels = 1
    test_num_classes = 10
    test_embed_dim = 128  # Smaller embed_dim for test
    test_depth = 3  # Fewer layers
    test_num_heads = 4  # Fewer heads

    model = ViTSmallForChars(
        img_size=test_img_size, patch_size=test_patch_size, in_channels=test_in_channels,
        num_classes=test_num_classes, embed_dim=test_embed_dim, depth=test_depth,
        num_heads=test_num_heads, use_class_token=True
    )
    print(f"ViTSmallForChars Model Structure (Layers: {test_depth}, Heads: {test_num_heads}, Embed: {test_embed_dim}):")
    # print(model) # Can be long

    dummy_batch_size = 2
    dummy_input = torch.randn(dummy_batch_size, test_in_channels, test_img_size[0], test_img_size[1])
    print(f"\nInput tensor shape: {dummy_input.shape}")

    logits = model(dummy_input)
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (dummy_batch_size, test_num_classes)

    # Test feature extraction
    feature_layer = "cls_token_embedding"
    model(dummy_input, extract_features_layer_names=[feature_layer, "encoder_block_1"])
    extracted_map = model.get_last_extracted_features()

    print(f"\nExtracted features for layers: {list(extracted_map.keys())}")
    if feature_layer in extracted_map:
        features = extracted_map[feature_layer]
        print(f"Features from '{feature_layer}' shape: {features.shape}")
        assert features.shape == (dummy_batch_size, test_embed_dim)

    if "encoder_block_1" in extracted_map:
        enc_block_features = extracted_map["encoder_block_1"]
        print(f"Features from 'encoder_block_1' shape: {enc_block_features.shape}")
        # Shape: (B, num_patches + 1, embed_dim)
        num_patches_expected = (test_img_size[0] // test_patch_size) * (test_img_size[1] // test_patch_size)
        assert enc_block_features.shape == (dummy_batch_size, num_patches_expected + 1, test_embed_dim)

    print("\nViTSmallForChars tests completed.")