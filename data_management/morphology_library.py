# ultimate_morph_generator/data_management/morphology_library.py
import sqlite3
import os
import uuid
import time
import json
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any
from threading import Lock

from ..config import get_config, DataManagementConfig
from ..utilities.type_definitions import MorphologySample, ImageHash, FeatureVector, ImagePath
from ..utilities.image_utils import standardize_image  # For saving consistent images
from ..utilities.logging_config import setup_logging

logger = setup_logging()  # 或者从主模块传递


class MorphologyLibraryDB:
    """
    Manages the SQLite database for storing morphology sample metadata.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = Lock()  # Thread-safe operations
        self._initialize_db()

    def _get_connection(self):
        """Establishes and returns a database connection."""
        # isolation_level=None for autocommit, or handle transactions explicitly
        return sqlite3.connect(self.db_path, timeout=10)  # Increased timeout

    def _initialize_db(self):
        """Creates the necessary tables if they don't exist."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS morphology_samples (
                    sample_id TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL UNIQUE,
                    image_hash TEXT NOT NULL, -- Stored as JSON string of list of ints
                    feature_vector_path TEXT, -- Path to .npy file

                    dgo_predicted_label INTEGER NOT NULL,
                    dgo_confidence REAL NOT NULL,
                    dgo_uncertainty REAL,

                    generation_step INTEGER NOT NULL,
                    parent_id TEXT, -- FK to self.sample_id (optional)
                    perturbation_applied TEXT,
                    perturbation_params_applied TEXT, -- Stored as JSON string

                    structure_check_passed INTEGER NOT NULL, -- 0 or 1 for boolean
                    novelty_score REAL,

                    creation_timestamp REAL NOT NULL,

                    FOREIGN KEY (parent_id) REFERENCES morphology_samples(sample_id)
                );
                """)
                # Indexes for faster queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_hash ON morphology_samples (image_hash);")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_generation_step ON morphology_samples (generation_step);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_dgo_confidence ON morphology_samples (dgo_confidence);")
                conn.commit()
                logger.info(f"Database initialized/verified at {self.db_path}")
            except sqlite3.Error as e:
                logger.error(f"Database initialization error: {e}")
                raise
            finally:
                conn.close()

    def add_sample(self, sample: MorphologySample) -> bool:
        """Adds a new sample to the database."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                INSERT INTO morphology_samples (
                    sample_id, image_path, image_hash, feature_vector_path,
                    dgo_predicted_label, dgo_confidence, dgo_uncertainty,
                    generation_step, parent_id, perturbation_applied, perturbation_params_applied,
                    structure_check_passed, novelty_score, creation_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample.sample_id,
                    sample.image_path,
                    json.dumps(sample.image_hash),  # Convert list to JSON string
                    sample.feature_vector_path,
                    sample.dgo_predicted_label,
                    sample.dgo_confidence,
                    sample.dgo_uncertainty,
                    sample.generation_step,
                    sample.parent_id,
                    sample.perturbation_applied,
                    json.dumps(sample.perturbation_params_applied) if sample.perturbation_params_applied else None,
                    1 if sample.structure_check_passed else 0,
                    sample.novelty_score,
                    sample.creation_timestamp
                ))
                conn.commit()
                logger.debug(f"Sample {sample.sample_id} added to DB.")
                return True
            except sqlite3.IntegrityError as e:  # E.g. UNIQUE constraint failed for image_path
                logger.warning(f"Failed to add sample {sample.sample_id} to DB (IntegrityError): {e}")
                return False
            except sqlite3.Error as e:
                logger.error(f"Error adding sample {sample.sample_id} to DB: {e}")
                conn.rollback()  # Rollback on other errors
                return False
            finally:
                conn.close()

    def get_sample_by_id(self, sample_id: str) -> Optional[MorphologySample]:
        """Retrieves a sample by its ID."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM morphology_samples WHERE sample_id = ?", (sample_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_morphologysample(row)
                return None
            except sqlite3.Error as e:
                logger.error(f"Error fetching sample {sample_id}: {e}")
                return None
            finally:
                conn.close()

    def get_all_hashes(self) -> List[ImageHash]:
        """Retrieves all image hashes from the library."""
        hashes = []
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT image_hash FROM morphology_samples")
                rows = cursor.fetchall()
                for row in rows:
                    hash_list = json.loads(row[0])  # JSON string to list
                    hashes.append(np.array(hash_list, dtype=np.uint8))
                return hashes
            except sqlite3.Error as e:
                logger.error(f"Error fetching all hashes: {e}")
                return []
            finally:
                conn.close()

    def get_all_samples_metadata(self, limit: Optional[int] = None, offset: int = 0) -> List[MorphologySample]:
        """Retrieves metadata for all samples, with optional limit and offset."""
        samples = []
        query = "SELECT * FROM morphology_samples ORDER BY creation_timestamp DESC"
        params = []
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                for row in rows:
                    samples.append(self._row_to_morphologysample(row))
                return samples
            except sqlite3.Error as e:
                logger.error(f"Error fetching all samples metadata: {e}")
                return []
            finally:
                conn.close()

    def get_sample_count(self) -> int:
        """Returns the total number of samples in the library."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM morphology_samples")
                count = cursor.fetchone()[0]
                return count
            except sqlite3.Error as e:
                logger.error(f"Error getting sample count: {e}")
                return 0
            finally:
                conn.close()

    def get_samples_for_dgo_finetuning(self, target_label: int, buffer_size: int,
                                       strategy: str = "recent_diverse") -> List[Tuple[ImagePath, int]]:
        """
        Selects samples for DGO finetuning.
        Strategies:
            - "recent_diverse": Mix of recent correct and incorrect (for target_label) samples.
            - "boundary_cases": Samples with confidence near decision boundary for target_label.
            - "high_low_confidence": Samples with very high or very low confidence for target_label.
        Returns list of (image_path, true_label_for_dgo) tuples.
        """
        samples_for_finetune = []
        # This is a simplified selection. A real implementation would be more complex.
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                # Example: Get recent samples, some predicted as target, some not
                # Positive samples (correctly identified as target, or ones we want to reinforce)
                # For simplicity, we take samples DGO thought were the target
                cursor.execute("""
                    SELECT image_path, dgo_predicted_label FROM morphology_samples 
                    WHERE dgo_predicted_label = ? 
                    ORDER BY creation_timestamp DESC LIMIT ?
                """, (target_label, buffer_size // 2))
                for row in cursor.fetchall():
                    samples_for_finetune.append((ImagePath(row[0]), row[1]))

                # Negative/Confusing samples (DGO thought was something else, or low confidence target)
                # Take samples that were NOT the target label, or were target but low confidence
                cursor.execute("""
                    SELECT image_path, dgo_predicted_label FROM morphology_samples
                    WHERE dgo_predicted_label != ? OR (dgo_predicted_label = ? AND dgo_confidence < 0.5)
                    ORDER BY RANDOM() LIMIT ? 
                """, (target_label, target_label, buffer_size - len(samples_for_finetune)))  # Fill remaining
                for row in cursor.fetchall():
                    # For negative samples, the "true_label_for_dgo" is what DGO predicted,
                    # or a special "not_target" label if DGO is binary.
                    # Here, we use DGO's prediction as the label.
                    samples_for_finetune.append((ImagePath(row[0]), row[1]))

                logger.info(
                    f"Selected {len(samples_for_finetune)} samples for DGO finetuning using '{strategy}' strategy.")
                return samples_for_finetune

            except sqlite3.Error as e:
                logger.error(f"Error fetching samples for DGO finetuning: {e}")
                return []
            finally:
                conn.close()

    def _row_to_morphologysample(self, row: tuple) -> MorphologySample:
        """Converts a database row tuple to a MorphologySample Pydantic model."""
        return MorphologySample(
            sample_id=row[0],
            image_path=ImagePath(row[1]),
            image_hash=json.loads(row[2]),  # Deserialize JSON string
            feature_vector_path=row[3],
            dgo_predicted_label=row[4],
            dgo_confidence=row[5],
            dgo_uncertainty=row[6],
            generation_step=row[7],
            parent_id=row[8],
            perturbation_applied=row[9],
            perturbation_params_applied=json.loads(row[10]) if row[10] else None,
            structure_check_passed=bool(row[11]),
            novelty_score=row[12],
            creation_timestamp=row[13]
        )


class MorphologyLibrary:
    """
    High-level interface for managing the morphology library,
    including image file storage and database interaction.
    """

    def __init__(self, char_string: str):
        self.config: DataManagementConfig = get_config().data_management
        self.system_config = get_config()  # For target image size etc.

        self.char_string = char_string
        self.base_dir = os.path.join(self.config.output_base_dir, f"char_{char_string}")
        self.image_dir = os.path.join(self.base_dir, self.config.image_archive_subfolder)
        self.feature_dir = os.path.join(self.base_dir, "feature_vectors")  # New folder for features
        self.db_path = os.path.join(self.base_dir, "database", self.config.database_filename)

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.db = MorphologyLibraryDB(self.db_path)
        logger.info(f"MorphologyLibrary for char '{char_string}' initialized. Base dir: {self.base_dir}")

    def add_new_sample(self,
                       image_data: np.ndarray,  # Raw perturbed image (CvImage uint8)
                       image_hash: ImageHash,
                       dgo_pred_label: int,
                       dgo_confidence: float,
                       generation_step: int,
                       structure_passed: bool,
                       feature_vector: Optional[FeatureVector] = None,
                       dgo_uncertainty: Optional[float] = None,
                       parent_id: Optional[str] = None,
                       perturb_name: Optional[str] = None,
                       perturb_params: Optional[Dict[str, Any]] = None,
                       novelty_score: Optional[float] = None
                       ) -> Optional[MorphologySample]:
        """
        Adds a new sample to the library. Saves image, (optionally) feature vector, and metadata.
        Returns the created MorphologySample object or None on failure.
        """
        sample_id = str(uuid.uuid4())

        # Standardize and save image
        # Image should be standardized before hashing and feature extraction for consistency
        # However, here we assume image_data is the final form to be saved.
        # For DGO input, it's preprocessed separately.
        # Let's standardize before saving to disk for visual consistency.

        # Use system_config for target_size and grayscale for standardization
        # This ensures images in the library are visually comparable.
        standardized_img_for_disk = standardize_image(
            image_data,
            target_size=self.system_config.data_management.target_image_size,
            grayscale=self.system_config.data_management.grayscale_input,
            invert_if_dark_on_light=True  # Common for character datasets
        )

        img_filename = f"{sample_id}{self.config.image_file_format}"
        img_path = os.path.join(self.image_dir, img_filename)

        try:
            # Ensure path exists (should be covered by __init__, but good practice)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            # OpenCV saves BGR by default. If grayscale, it's fine.
            # If color, and standardized_img_for_disk is RGB, convert to BGR.
            # standardize_image currently returns grayscale or original format.
            if standardized_img_for_disk.ndim == 3 and standardized_img_for_disk.shape[2] == 3:
                # Assuming standardize_image might return RGB if it started as color and grayscale=False
                # cv2.imwrite expects BGR if color.
                # For now, assume standardize_image outputs what cv2.imwrite expects or grayscale.
                pass

            save_success = cv2.imwrite(img_path, standardized_img_for_disk)
            if not save_success:
                logger.error(f"Failed to save image to {img_path}")
                return None
        except Exception as e:
            logger.error(f"Exception saving image {img_path}: {e}")
            return None

        # Save feature vector (optional)
        fv_path: Optional[str] = None
        if feature_vector is not None:
            fv_filename = f"{sample_id}_features.npy"
            fv_path = os.path.join(self.feature_dir, fv_filename)
            try:
                os.makedirs(os.path.dirname(fv_path), exist_ok=True)
                np.save(fv_path, feature_vector)
            except Exception as e:
                logger.error(f"Exception saving feature vector {fv_path}: {e}")
                fv_path = None  # Don't record path if save failed

        # Create MorphologySample Pydantic model
        sample_entry = MorphologySample(
            sample_id=sample_id,
            image_path=ImagePath(os.path.relpath(img_path, self.base_dir)),  # Store relative path
            image_hash=image_hash.tolist(),  # np.array to list for JSON
            feature_vector_path=os.path.relpath(fv_path, self.base_dir) if fv_path else None,
            dgo_predicted_label=dgo_pred_label,
            dgo_confidence=dgo_confidence,
            dgo_uncertainty=dgo_uncertainty,
            generation_step=generation_step,
            parent_id=parent_id,
            perturbation_applied=perturb_name,
            perturbation_params_applied=perturb_params,
            structure_check_passed=structure_passed,
            novelty_score=novelty_score,
            creation_timestamp=time.time()
        )

        if self.db.add_sample(sample_entry):
            logger.info(
                f"New sample {sample_id} (Gen: {generation_step}, Label: {dgo_pred_label}, Conf: {dgo_confidence:.3f}) added to library.")
            return sample_entry
        else:
            # If DB add fails, try to clean up saved files
            if os.path.exists(img_path): os.remove(img_path)
            if fv_path and os.path.exists(fv_path): os.remove(fv_path)
            logger.warning(f"Failed to add sample {sample_id} to DB, file artifacts cleaned up.")
            return None

    def get_sample_image(self, sample_id_or_path: Union[str, ImagePath]) -> Optional[np.ndarray]:
        """Loads the image for a given sample ID or relative path."""
        img_path_to_load: Optional[str] = None
        if "/" in sample_id_or_path or "\\" in sample_id_or_path:  # Likely a path
            img_path_to_load = os.path.join(self.base_dir, sample_id_or_path)
        else:  # Assume it's an ID
            sample_meta = self.db.get_sample_by_id(sample_id_or_path)
            if sample_meta:
                img_path_to_load = os.path.join(self.base_dir, sample_meta.image_path)

        if img_path_to_load and os.path.exists(img_path_to_load):
            # Load as grayscale as that's how they are typically processed by DGO
            # However, the saved image might be color if grayscale_input config is False
            # For consistency, let's adhere to the grayscale_input config.
            if self.system_config.data_management.grayscale_input:
                img = cv2.imread(img_path_to_load, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path_to_load, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning(f"Failed to load image from {img_path_to_load} (cv2.imread returned None).")
            return img
        else:
            logger.warning(f"Image not found for {sample_id_or_path} at resolved path {img_path_to_load}.")
            return None

    def get_sample_feature_vector(self, sample_id: str) -> Optional[FeatureVector]:
        """Loads the feature vector for a given sample ID."""
        sample_meta = self.db.get_sample_by_id(sample_id)
        if sample_meta and sample_meta.feature_vector_path:
            fv_path_abs = os.path.join(self.base_dir, sample_meta.feature_vector_path)
            if os.path.exists(fv_path_abs):
                try:
                    return np.load(fv_path_abs)
                except Exception as e:
                    logger.error(f"Error loading feature vector {fv_path_abs}: {e}")
                    return None
        logger.warning(f"Feature vector not found or not recorded for sample {sample_id}.")
        return None

    def get_all_hashes(self) -> List[ImageHash]:
        return self.db.get_all_hashes()

    def get_all_samples_metadata(self, limit: Optional[int] = None, offset: int = 0) -> List[MorphologySample]:
        return self.db.get_all_samples_metadata(limit=limit, offset=offset)

    def get_count(self) -> int:
        return self.db.get_sample_count()

    def get_samples_for_dgo_finetuning(self, target_label: int, buffer_size: int,
                                       strategy: str = "recent_diverse") -> List[Tuple[np.ndarray, int]]:
        """
        Retrieves image data and labels for DGO finetuning.
        Returns list of (image_numpy_array, label)
        """
        metadata_tuples = self.db.get_samples_for_dgo_finetuning(target_label, buffer_size, strategy)

        loaded_samples = []
        for img_relative_path, label in metadata_tuples:
            img_data = self.get_sample_image(img_relative_path)
            if img_data is not None:
                # Ensure image is in the format DGO expects for training (e.g., 28x28 grayscale)
                # DGO's preprocess_image_for_dgo handles PIL/path to tensor.
                # Here we need np.array for DGO.finetune which might expect np array directly.
                # For simplicity, assume get_sample_image already returns it in a usable raw form.
                # The DGO.finetune method will handle its own preprocessing from np.ndarray.
                loaded_samples.append((img_data, label))
            else:
                logger.warning(f"Could not load image {img_relative_path} for DGO finetuning.")
        return loaded_samples


if __name__ == "__main__":
    # --- Test MorphologyLibrary ---
    # Setup a temporary config for testing
    from ..config import SystemConfig

    temp_cfg_data = {
        "project_name": "MorphLibTest",
        "target_character_string": "test_char",
        "data_management": {
            "output_base_dir": "./temp_morph_lib_output/",
            "target_image_size": (32, 32),  # Test with different size
            "grayscale_input": True
        },
        "logging": {"level": "DEBUG"}  # Enable debug for more verbose output
    }
    # Override global config for this test script
    from ..config import _config_instance

    _config_instance = SystemConfig.model_validate(temp_cfg_data)

    logger = setup_logging()  # Re-init logger with new config

    test_char_str = get_config().target_character_string
    library = MorphologyLibrary(char_string=test_char_str)

    print(f"Library count: {library.get_count()}")

    # Create a dummy image and features
    dummy_image = np.random.randint(0, 256, (40, 40), dtype=np.uint8)  # Raw image
    cv2.putText(dummy_image, "T", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

    dummy_hash = np.random.randint(0, 2, size=64, dtype=np.uint8)
    dummy_features = np.random.rand(128).astype(np.float32)

    # Add a sample
    added_sample = library.add_new_sample(
        image_data=dummy_image,
        image_hash=dummy_hash,
        dgo_pred_label=int(test_char_str) if test_char_str.isdigit() else 0,
        dgo_confidence=0.95,
        generation_step=1,
        structure_passed=True,
        feature_vector=dummy_features,
        dgo_uncertainty=0.05,
        parent_id=None,
        perturb_name="initial",
        novelty_score=1.0
    )

    if added_sample:
        print(f"Added sample: {added_sample.sample_id}")
        print(f"Library count after add: {library.get_count()}")

        # Retrieve and verify
        retrieved_meta = library.db.get_sample_by_id(added_sample.sample_id)
        assert retrieved_meta is not None
        assert retrieved_meta.dgo_confidence == 0.95
        print(f"Retrieved metadata for {retrieved_meta.sample_id}")

        retrieved_image = library.get_sample_image(added_sample.sample_id)
        assert retrieved_image is not None
        # Saved image will be standardized to (32,32) as per config
        assert retrieved_image.shape == get_config().data_management.target_image_size
        print(f"Retrieved image shape: {retrieved_image.shape}")
        # cv2.imshow("Retrieved", retrieved_image); cv2.waitKey(0); cv2.destroyAllWindows()

        retrieved_features = library.get_sample_feature_vector(added_sample.sample_id)
        assert retrieved_features is not None
        assert np.array_equal(retrieved_features, dummy_features)
        print(f"Retrieved features match.")

        all_hashes = library.get_all_hashes()
        assert len(all_hashes) == 1
        assert np.array_equal(all_hashes[0], dummy_hash)
        print("Hashes retrieved correctly.")

        all_meta = library.get_all_samples_metadata()
        assert len(all_meta) == 1
        print(f"All metadata retrieved: {len(all_meta)} samples.")

        finetune_data = library.get_samples_for_dgo_finetuning(target_label=0, buffer_size=10)
        print(f"Finetuning data (image, label) pairs: {len(finetune_data)}")
        if finetune_data:
            img_sample, label_sample = finetune_data[0]
            print(f"  Example finetune sample image shape: {img_sample.shape}, label: {label_sample}")


    else:
        print("Failed to add sample.")

    print("\nMorphologyLibrary tests completed.")
    # Consider cleaning up temp_morph_lib_output/ after tests
    # import shutil
    # shutil.rmtree("./temp_morph_lib_output/", ignore_errors=True)