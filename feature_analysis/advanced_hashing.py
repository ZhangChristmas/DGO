# ultimate_morph_generator/feature_analysis/advanced_hashing.py
import numpy as np
import torch  # For potential learned hashing models
import torch.nn as nn
from typing import Optional, Union, List, Tuple

from ..config import get_config, FeatureAnalysisConfig
from ..utilities.type_definitions import FeatureVector, ImageHash
from ..utilities.logging_config import setup_logging

logger = setup_logging()


# --- Learned Hashing Model (Example Stub) ---
class LearnedHashingNet(nn.Module):
    """
    A small neural network that learns to project feature vectors into a binary hash space.
    This is an advanced concept and requires a separate training phase for the hasher itself.
    (e.g., using siamese networks with triplet loss on (anchor, positive, negative) feature hashes)
    """

    def __init__(self, feature_dim: int, hash_length: int):
        super(LearnedHashingNet, self).__init__()
        self.feature_dim = feature_dim
        self.hash_length = hash_length

        # Example layers: can be more complex
        self.fc1 = nn.Linear(feature_dim, (feature_dim + hash_length) // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear((feature_dim + hash_length) // 2, hash_length)
        # The output of fc2 will be passed through a tanh (to get values in [-1, 1])
        # and then thresholded at 0 to get binary {0, 1} or {-1, 1}.
        # Or, use a sigmoid and threshold at 0.5.
        # For direct binary output during training, one might use Gumbel-Softmax or straight-through estimator.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # For training, might output logits for a BCEWithLogitsLoss, or use tanh for hash-like loss
        # For inference (hashing):
        return torch.tanh(x)  # Output in [-1, 1]

    def get_binary_hash(self, features_tensor: torch.Tensor) -> torch.Tensor:
        self.eval()  # Ensure eval mode for hashing
        with torch.no_grad():
            hash_activations = self.forward(features_tensor)  # Values in [-1, 1]
            binary_hash = (hash_activations > 0).type(torch.uint8)  # {0, 1}
        return binary_hash


class AdvancedFeatureHasher:
    """
    Provides various methods to hash DGO feature vectors for novelty assessment.
    """

    def __init__(self, feature_dim: int,
                 hashing_cfg: FeatureAnalysisConfig,  # Contains hash_length, method etc.
                 device: Optional[torch.device] = None):
        self.feature_dim = feature_dim
        self.cfg = hashing_cfg
        self.hash_length = self.cfg.hash_length
        self.method = self.cfg.hashing_method
        self.device = device if device else torch.device("cpu")  # Learned hasher might use GPU

        self._projection_matrix: Optional[np.ndarray] = None
        self._learned_hasher_model: Optional[LearnedHashingNet] = None
        self._multi_scale_thresholds: Optional[List[float]] = None

        self._initialize_hasher()
        logger.info(
            f"AdvancedFeatureHasher initialized with method '{self.method}', feature_dim={feature_dim}, hash_length={self.hash_length}.")

    def _initialize_hasher(self):
        """Initializes internal structures based on the chosen hashing method."""
        if self.method == "simhash_projection":
            # Create a stable random projection matrix
            # (feature_dim, hash_length)
            # Seed for reproducibility if needed, but usually want random projection
            # np.random.seed(get_config().random_seed) # Optional, if global seed affects this
            self._projection_matrix = np.random.randn(self.feature_dim, self.hash_length)

        elif self.method == "thresholding":
            # Simple thresholding, often on a subset of features or PCA components.
            # If using a subset, hash_length should be <= feature_dim.
            if self.hash_length > self.feature_dim:
                logger.warning(
                    f"Thresholding hasher: hash_length ({self.hash_length}) > feature_dim ({self.feature_dim}). "
                    f"Will use full feature_dim if no specific selection strategy.")
                # self.hash_length = self.feature_dim # Or error out
            # No specific matrix needed here, logic is in get_hash.

        elif self.method == "multi_scale_concat":
            # Example: Split hash_length among different thresholding strategies or projections
            # This needs more specific configuration. For now, it's a placeholder.
            # Could involve different projection matrices for parts of the hash.
            # Or different thresholding levels on the same features.
            logger.warning("Multi-scale_concat hashing method is conceptual. Using SimHash projection as fallback.")
            self.method = "simhash_projection"  # Fallback
            self._projection_matrix = np.random.randn(self.feature_dim, self.hash_length)

        elif self.method == "learned_hash":
            # Load or initialize the learned hashing model
            self._learned_hasher_model = LearnedHashingNet(self.feature_dim, self.hash_length).to(self.device)
            # Path to pretrained learned hasher model should be in config
            learned_hasher_path = getattr(self.cfg, 'learned_hasher_model_path', None)
            if learned_hasher_path and os.path.exists(learned_hasher_path):
                try:
                    self._learned_hasher_model.load_state_dict(
                        torch.load(learned_hasher_path, map_location=self.device))
                    logger.info(f"Learned hashing model loaded from {learned_hasher_path}")
                except Exception as e:
                    logger.error(
                        f"Failed to load learned hashing model from {learned_hasher_path}: {e}. Using random init.")
            else:
                logger.warning(
                    f"Learned hashing model path not provided or invalid. Using random initialized hasher network (needs training).")
            self._learned_hasher_model.eval()
        else:
            raise ValueError(f"Unknown hashing method specified: {self.method}")

    def get_hash(self, feature_vector: FeatureVector) -> ImageHash:
        """
        Computes a binary hash for the given DGO feature vector.
        feature_vector: 1D NumPy array of floats.
        Returns: 1D NumPy array of uint8 (0s and 1s) representing the hash.
        """
        if not isinstance(feature_vector, np.ndarray):
            raise TypeError("Feature vector must be a NumPy array.")
        if feature_vector.ndim != 1:
            # Allow (1, D) shape and squeeze it
            if feature_vector.ndim == 2 and feature_vector.shape[0] == 1:
                feature_vector = feature_vector.squeeze(0)
            else:
                raise ValueError(f"Feature vector must be 1D (or squeezable to 1D), got shape {feature_vector.shape}")

        if feature_vector.shape[0] != self.feature_dim:
            # Handle dimension mismatch (e.g. DGO model changed)
            # Option 1: Error out. Option 2: Pad/truncate (can be problematic).
            # logger.warning(f"Feature vector dim {feature_vector.shape[0]} != expected {self.feature_dim}. Attempting to adapt...")
            # Simple adaptation: if smaller, pad with zeros; if larger, truncate.
            if feature_vector.shape[0] < self.feature_dim:
                padded_vector = np.zeros(self.feature_dim, dtype=feature_vector.dtype)
                padded_vector[:feature_vector.shape[0]] = feature_vector
                feature_vector = padded_vector
            else:
                feature_vector = feature_vector[:self.feature_dim]

        binary_hash_np: Optional[np.ndarray] = None

        if self.method == "simhash_projection":
            if self._projection_matrix is None:  # Should be initialized
                raise RuntimeError("SimHash projection matrix not initialized.")
            # Projected: (1, D) @ (D, H) -> (1, H)
            projected_values = np.dot(feature_vector, self._projection_matrix)
            binary_hash_np = (projected_values > 0).astype(np.uint8)

        elif self.method == "thresholding":
            # Example: Threshold first `hash_length` features against their median or mean.
            # This is a very basic form.
            if self.hash_length > feature_vector.shape[0]:
                # This case should be handled by config validation or init
                target_features = feature_vector
            else:
                target_features = feature_vector[:self.hash_length]

            threshold_val = np.median(target_features)  # Or np.mean, or 0.0
            binary_hash_np = (target_features > threshold_val).astype(np.uint8)
            # Ensure hash has `self.hash_length` (might be shorter if feature_vector was shorter)
            if binary_hash_np.shape[0] < self.hash_length:
                padded_hash = np.zeros(self.hash_length, dtype=np.uint8)
                padded_hash[:binary_hash_np.shape[0]] = binary_hash_np
                binary_hash_np = padded_hash

        elif self.method == "learned_hash":
            if self._learned_hasher_model is None:
                raise RuntimeError("Learned hashing model not initialized.")

            # Convert numpy feature vector to torch tensor
            features_tensor = torch.from_numpy(feature_vector).float().unsqueeze(0).to(self.device)  # (1, D)
            hash_tensor_uint8 = self._learned_hasher_model.get_binary_hash(features_tensor)  # (1, H) uint8
            binary_hash_np = hash_tensor_uint8.squeeze(0).cpu().numpy()

        else:  # Fallback or error for unhandled methods (should be caught in init)
            raise NotImplementedError(f"Hashing method '{self.method}' processing not fully implemented in get_hash.")

        if binary_hash_np is None:  # Should not happen if logic is correct
            logger.error(f"Hash generation failed for method {self.method}. Returning zero hash.")
            return np.zeros(self.hash_length, dtype=np.uint8)

        # Final check for hash length consistency (e.g. if thresholding on short vector)
        if binary_hash_np.shape[0] != self.hash_length:
            # This might indicate an issue with the method's logic or config
            logger.warning(f"Generated hash length {binary_hash_np.shape[0]} "
                           f"differs from target {self.hash_length} for method {self.method}. Adjusting.")
            # Simple adjustment: pad with zeros or truncate
            final_hash = np.zeros(self.hash_length, dtype=np.uint8)
            len_to_copy = min(binary_hash_np.shape[0], self.hash_length)
            final_hash[:len_to_copy] = binary_hash_np[:len_to_copy]
            return ImageHash(final_hash)  # Cast to NewType

        return ImageHash(binary_hash_np)

    @staticmethod
    def hamming_distance(hash1: ImageHash, hash2: ImageHash) -> int:
        """Computes the Hamming distance between two binary hashes."""
        if not isinstance(hash1, np.ndarray) or not isinstance(hash2, np.ndarray):
            raise TypeError("Hashes must be NumPy arrays for Hamming distance calculation.")
        if hash1.shape != hash2.shape:
            # This can happen if hash_length changes or there's an error.
            # Return max possible distance as a penalty, or handle as error.
            logger.warning(f"Attempting Hamming distance on hashes of different shapes: "
                           f"{hash1.shape} vs {hash2.shape}. Returning max possible distance.")
            return max(hash1.shape[0], hash2.shape[0])

            # Ensure they are binary (0 or 1) for XOR interpretation, though (h1 != h2).sum() works generally.
        return np.sum(hash1 != hash2)

    def needs_retraining_for_hasher(self) -> bool:
        """Indicates if the chosen hashing method (e.g., learned_hash) requires its own training phase."""
        return self.method == "learned_hash"

    def train_learned_hasher(self, training_data: List[Tuple[FeatureVector, FeatureVector, FeatureVector]],
                             epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):
        """
        Trains the learned hashing model.
        training_data: List of (anchor_features, positive_features, negative_features) tuples.
        This is a conceptual training loop.
        """
        if not self.method == "learned_hash" or self._learned_hasher_model is None:
            logger.warning("train_learned_hasher called, but method is not 'learned_hash' or model not init.")
            return

        logger.info(f"Starting training for learned hasher ({epochs} epochs)...")
        self._learned_hasher_model.train()  # Set to training mode
        optimizer = torch.optim.Adam(self._learned_hasher_model.parameters(), lr=lr)

        # Example: Triplet loss for hashing
        # L(A,P,N) = max( ||h(A) - h(P)||^2 - ||h(A) - h(N)||^2 + margin, 0 )
        # We want hash of A and P to be close, hash of A and N to be far.
        # Here, "close/far" means Hamming distance for binary hashes.
        # For continuous outputs before binarization (e.g., tanh layer), can use Euclidean/cosine.
        # Training binary hash nets is non-trivial due to non-differentiable binarization.
        # Common approach: relax binarization during training (e.g. use tanh outputs directly in loss)
        # or use techniques like Straight-Through Estimator.

        # Simplified loss for demonstration:
        # Try to make tanh outputs for (A,P) similar and (A,N) different using MSE-like terms.
        # This is NOT a standard hashing loss, just for concept.

        # A proper triplet loss for continuous embeddings h_A, h_P, h_N:
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)  # p=2 for L2 distance

        for epoch in range(epochs):
            np.random.shuffle(training_data)  # Shuffle data each epoch
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, len(training_data), batch_size):
                batch_triplets = training_data[i: i + batch_size]
                if not batch_triplets: continue

                # Collate batch
                anchors_np = np.array([t[0] for t in batch_triplets])
                positives_np = np.array([t[1] for t in batch_triplets])
                negatives_np = np.array([t[2] for t in batch_triplets])

                anchors_t = torch.from_numpy(anchors_np).float().to(self.device)
                positives_t = torch.from_numpy(positives_np).float().to(self.device)
                negatives_t = torch.from_numpy(negatives_np).float().to(self.device)

                optimizer.zero_grad()

                # Get continuous embeddings (before binarization) from the hasher model
                h_anchors = self._learned_hasher_model(anchors_t)  # tanh outputs
                h_positives = self._learned_hasher_model(positives_t)
                h_negatives = self._learned_hasher_model(negatives_t)

                loss = triplet_loss_fn(h_anchors, h_positives, h_negatives)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Learned Hasher Training - Epoch {epoch + 1}/{epochs}, Avg Triplet Loss: {avg_epoch_loss:.4f}")

        self._learned_hasher_model.eval()  # Back to eval mode
        logger.info("Learned hasher training finished.")
        # Optionally save the trained hasher model
        # model_save_path = getattr(self.cfg, 'learned_hasher_model_path', './learned_hasher.pth')
        # torch.save(self._learned_hasher_model.state_dict(), model_save_path)


if __name__ == "__main__":
    # --- Test AdvancedFeatureHasher ---
    from ..config import SystemConfig

    test_feature_dim = 128
    test_hash_length = 64

    # Config for SimHash projection
    temp_sys_cfg_data_simhash = {
        "feature_analysis": {
            "hashing_method": "simhash_projection",
            "hash_length": test_hash_length,
            # novelty_hamming_distance_threshold_ratio etc are not used by hasher itself
        },
        "logging": {"level": "DEBUG"}
    }
    from ..config import _config_instance

    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_simhash)
    cfg_simhash = get_config().feature_analysis

    print("\n--- Testing SimHash Projection Hasher ---")
    hasher_simhash = AdvancedFeatureHasher(feature_dim=test_feature_dim, hashing_cfg=cfg_simhash)

    fv1 = np.random.randn(test_feature_dim).astype(np.float32)
    fv2 = np.random.randn(test_feature_dim).astype(np.float32)
    fv3 = fv1 + 0.01 * np.random.randn(test_feature_dim).astype(np.float32)  # Similar to fv1

    hash1_s = hasher_simhash.get_hash(fv1)
    hash2_s = hasher_simhash.get_hash(fv2)
    hash3_s = hasher_simhash.get_hash(fv3)

    print(f"Hash1 (SimHash) length: {hash1_s.shape[0]}, example: {hash1_s[:10]}")
    assert hash1_s.shape[0] == test_hash_length
    assert hash1_s.dtype == np.uint8

    dist12_s = AdvancedFeatureHasher.hamming_distance(hash1_s, hash2_s)
    dist13_s = AdvancedFeatureHasher.hamming_distance(hash1_s, hash3_s)
    print(f"SimHash Hamming distances: d(h1,h2)={dist12_s}, d(h1,h3)={dist13_s}")
    assert dist13_s < dist12_s or dist12_s > test_hash_length * 0.3  # Expect similar to be closer, random to be further

    # Config for Learned Hashing (conceptual test, no real training data)
    temp_sys_cfg_data_learned = {
        "feature_analysis": {
            "hashing_method": "learned_hash",
            "hash_length": test_hash_length,
            "learned_hasher_model_path": None  # No pretrained model for this test
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_learned)  # Update global config
    cfg_learned = get_config().feature_analysis

    print("\n--- Testing Learned Hasher (Untrained) ---")
    hasher_learned = AdvancedFeatureHasher(feature_dim=test_feature_dim, hashing_cfg=cfg_learned)
    assert hasher_learned.needs_retraining_for_hasher()

    hash1_l = hasher_learned.get_hash(fv1)
    print(f"Hash1 (Learned, untrained) length: {hash1_l.shape[0]}, example: {hash1_l[:10]}")
    assert hash1_l.shape[0] == test_hash_length

    # Conceptual: Test training the learned hasher (if it were implemented more fully)
    # Create dummy triplet data for hasher training
    # num_triplets = 20
    # dummy_triplet_data = []
    # for _ in range(num_triplets):
    #     anchor = np.random.randn(test_feature_dim).astype(np.float32)
    #     positive = anchor + 0.1 * np.random.randn(test_feature_dim).astype(np.float32)
    #     negative = np.random.randn(test_feature_dim).astype(np.float32)
    #     dummy_triplet_data.append((anchor, positive, negative))
    # print(f"Training learned hasher with {len(dummy_triplet_data)} dummy triplets...")
    # hasher_learned.train_learned_hasher(dummy_triplet_data, epochs=1, batch_size=4)
    # print("Learned hasher conceptual training finished.")
    # hash1_l_trained = hasher_learned.get_hash(fv1) # Hash after "training"
    # hash3_l_trained = hasher_learned.get_hash(fv3) # fv3 is similar to fv1
    # dist13_l_trained = AdvancedFeatureHasher.hamming_distance(hash1_l_trained, hash3_l_trained)
    # print(f"Learned Hasher (after dummy train) d(h1,h3) = {dist13_l_trained}. Should be small if training worked.")

    print("\nAdvancedFeatureHasher tests completed.")