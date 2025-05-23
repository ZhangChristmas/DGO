# ultimate_morph_generator/core_logic/generation_manager.py
import os
import time
import random

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

from ..config import get_config, GenerationManagerConfig, PerturbationSuiteConfig, FeatureAnalysisConfig
from ..utilities.type_definitions import CvImage, MorphologySample, FeatureVector, ImageHash, ImagePath
from ..utilities.logging_config import setup_logging
from ..dgo_oracle.dgo_model_handler import DGOModelHandler
from ..perturbation_suite.perturbation_orchestrator import PerturbationOrchestrator
from ..structure_guard import StructureGuard
from ..feature_analysis.advanced_hashing import AdvancedFeatureHasher
from ..feature_analysis.diversity_metrics import LibraryDiversityAnalyzer  # For parent selection based on density
from ..data_management.morphology_library import MorphologyLibrary
from ..data_management.dataset_utils import load_initial_char_samples  # For initial seeds
from .quality_assessor import CandidateQualityAssessor

logger = setup_logging()


class GenerationManager:
    """
    Manages the iterative process of generating novel character morphologies.
    - Selects parent samples.
    - Applies perturbations.
    - Assesses candidates using QualityAssessor.
    - Adds qualifying samples to the MorphologyLibrary.
    - Triggers DGO finetuning.
    - Handles stopping conditions.
    """

    def __init__(self,
                 cfg_system=get_config()  # Allow passing a specific config for testing
                 ):
        self.cfg = cfg_system
        self.gen_cfg: GenerationManagerConfig = self.cfg.generation_manager

        self.target_char_idx: int = self.cfg.target_character_index
        self.target_char_string: str = self.cfg.target_character_string

        self.device = self.cfg.get_actual_device()

        # Initialize core components
        logger.info("Initializing GenerationManager components...")
        self.dgo_handler = DGOModelHandler(self.cfg.dgo_oracle, self.cfg.data_management, self.device)

        # For StructureGuard, pre-compute reference PH signature if PH is enabled
        ref_ph_sig = None
        if self.cfg.structure_guard.advanced_topology.enabled and \
                self.cfg.structure_guard.advanced_topology.persistent_homology_params.get("enabled", True):
            # Need a path to a clean reference image of the target character
            # This path should be ideally in config or discoverable.
            # For now, assume initial samples path contains one.
            initial_paths, _ = load_initial_char_samples(self.target_char_string, self.target_char_idx, max_samples=1)
            if initial_paths:
                from ..structure_guard import compute_reference_ph_signature_for_char  # Local import
                ref_ph_sig = compute_reference_ph_signature_for_char(
                    initial_paths[0], self.cfg.structure_guard.advanced_topology
                )
            else:
                logger.warning("No initial sample found to compute reference PH signature for StructureGuard.")

        self.structure_checker = StructureGuard(
            self.cfg.structure_guard, self.target_char_string, reference_ph_signature=ref_ph_sig
        )

        # Infer feature_dim for hasher and analyzer
        self.feature_dim = self.dgo_handler.get_feature_dimensionality()
        if self.feature_dim is None:
            # Fallback or raise error if feature_dim is crucial and cannot be inferred
            logger.warning("Could not infer DGO feature dimensionality. Using default from config or 128.")
            self.feature_dim = self.cfg.feature_analysis.get('fallback_feature_dim',
                                                             128)  # Add to FeatureAnalysisConfig if needed

        self.feature_hasher = AdvancedFeatureHasher(self.feature_dim, self.cfg.feature_analysis, self.device)

        self.library_analyzer: Optional[LibraryDiversityAnalyzer] = None
        if self.cfg.feature_analysis.diversity_strategy != "novelty_only":  # If more advanced diversity needed
            self.library_analyzer = LibraryDiversityAnalyzer(self.feature_dim, self.cfg.feature_analysis)

        self.perturb_orchestrator = PerturbationOrchestrator(
            self.cfg.perturbation_suite, self.cfg.data_management.target_image_size, self.dgo_handler
        )

        self.morph_library = MorphologyLibrary(char_string=self.target_char_string)

        self.quality_assessor = CandidateQualityAssessor(
            target_char_idx=self.target_char_idx,
            gen_cfg=self.gen_cfg,
            dgo_handler=self.dgo_handler,
            structure_checker=self.structure_checker,
            feature_hasher=self.feature_hasher,
            library_analyzer=self.library_analyzer,
            morphology_library_hashes=self.morph_library.get_all_hashes(),  # Initial state
            morphology_library_features=self._get_all_library_features()  # Initial state
        )

        # State variables for the generation loop
        self.generation_step: int = 0
        self.consecutive_no_new_samples: int = 0
        self.last_dgo_finetune_library_size: int = 0

        # Buffer for (image_for_finetune, label) pairs, where image is CvImage np.ndarray
        self.dgo_finetune_data_buffer = deque(maxlen=self.gen_cfg.dgo_finetune_data_buffer_size)

        self.initial_seed_samples: List[CvImage] = self._load_initial_seed_samples()
        if not self.initial_seed_samples:
            logger.error("CRITICAL: No initial seed samples loaded. Generation cannot proceed without seeds.")
            raise ValueError("No initial seed samples available.")

    def _load_initial_seed_samples(self) -> List[CvImage]:
        """Loads initial clean samples of the target character to seed the generation."""
        # Use dataset_utils to load paths, then load images into CvImage format
        # These are used as the very first parents.
        paths, _ = load_initial_char_samples(self.target_char_string, self.target_char_idx,
                                             max_samples=getattr(self.gen_cfg, 'num_initial_seeds',
                                                                 10))  # Add to GenManagerConfig

        cv_images: List[CvImage] = []
        for p_str in paths:
            img = cv2.imread(str(p_str), cv2.IMREAD_UNCHANGED)  # Load as is initially
            if img is not None:
                # Standardize to consistent format (e.g. grayscale, target size) expected by perturbers
                from ..utilities.image_utils import standardize_image  # Local import
                std_img = standardize_image(img,
                                            target_size=self.cfg.data_management.target_image_size,
                                            grayscale=self.cfg.data_management.grayscale_input,
                                            invert_if_dark_on_light=True)
                cv_images.append(std_img)
            else:
                logger.warning(f"Failed to load initial seed image: {p_str}")

        logger.info(f"Loaded {len(cv_images)} initial seed samples.")
        return cv_images

    def _get_all_library_features(self) -> List[FeatureVector]:
        """Helper to load all feature vectors from the library (if stored)."""
        # This could be slow if features are large and many.
        # Optimization: store features in memory if feasible, or sample.
        all_meta = self.morph_library.get_all_samples_metadata()
        features_list = []
        for meta_sample in all_meta:
            if meta_sample.feature_vector_path:
                fv = self.morph_library.get_sample_feature_vector(meta_sample.sample_id)
                if fv is not None:
                    features_list.append(fv)
        return features_list

    def _select_parent_sample(self) -> CvImage:
        """
        Selects a parent sample for perturbation.
        Strategy can be configured (random, high-confidence, boundary-cases, low-density).
        """
        strategy = self.gen_cfg.parent_selection_strategy

        # Fallback to initial seeds if library is small or strategy demands it
        use_initial_seed_prob = getattr(self.gen_cfg, 'parent_selection_use_initial_seed_prob', 0.1)
        if not self.morph_library.get_count() or \
                (self.initial_seed_samples and random.random() < use_initial_seed_prob) or \
                (self.morph_library.get_count() < getattr(self.gen_cfg,
                                                          'min_library_size_for_advanced_parent_selection',
                                                          10)):  # Add to GenCfg
            logger.debug("Selecting parent from initial seed samples.")
            return random.choice(self.initial_seed_samples).copy()

        # Retrieve candidate parents from library (e.g., all or a subset)
        # This could be costly if library is huge. Sample if needed.
        # Max parents to consider for selection:
        max_parents_to_consider = getattr(self.gen_cfg, 'max_parents_to_consider_for_selection', 200)
        parent_candidates_meta = self.morph_library.get_all_samples_metadata(limit=max_parents_to_consider)

        if not parent_candidates_meta:  # Should not happen if library has items
            logger.warning("No parent candidates found in library despite having items. Fallback to initial seeds.")
            return random.choice(self.initial_seed_samples).copy()

        selected_meta: Optional[MorphologySample] = None

        if strategy == "random":
            selected_meta = random.choice(parent_candidates_meta)

        elif strategy == "high_confidence":
            # Sort by DGO confidence descending, pick from top N
            parent_candidates_meta.sort(key=lambda s: s.dgo_confidence, reverse=True)
            selected_meta = random.choice(parent_candidates_meta[:max(1, len(parent_candidates_meta) // 5)])  # Top 20%

        elif strategy == "boundary_cases":
            # Samples with confidence near the DGO acceptance threshold or other defined boundaries
            boundary_low, boundary_high = self.gen_cfg.dgo_boundary_case_confidence_range
            boundary_samples = [s for s in parent_candidates_meta if
                                boundary_low <= s.dgo_confidence < boundary_high and s.dgo_predicted_label == self.target_char_idx]
            if boundary_samples:
                selected_meta = random.choice(boundary_samples)
            else:  # Fallback if no boundary cases found in current candidates
                logger.debug("No boundary-case parents found, falling back to random parent selection.")
                selected_meta = random.choice(parent_candidates_meta)

        elif strategy == "low_density_feature" and self.library_analyzer and self.library_analyzer.kde_estimator:
            # Select parent from a low-density region in the feature space
            # This requires features for all parent_candidates_meta
            candidate_features = []
            valid_meta_for_density = []
            for meta in parent_candidates_meta:
                fv = self.morph_library.get_sample_feature_vector(meta.sample_id)
                if fv is not None:
                    candidate_features.append(fv)
                    valid_meta_for_density.append(meta)

            if candidate_features:
                densities = self.library_analyzer.estimate_feature_space_density_kde(
                    feature_vectors=self.quality_assessor.current_library_features,  # Fit on whole library
                    query_points=candidate_features
                )
                if densities is not None and len(densities) > 0:
                    # Select sample with the minimum density (or among bottom N percent)
                    min_density_idx = np.argmin(densities)
                    selected_meta = valid_meta_for_density[min_density_idx]
                else:  # Fallback
                    selected_meta = random.choice(parent_candidates_meta)
            else:  # Fallback
                selected_meta = random.choice(parent_candidates_meta)

        elif strategy == "high_uncertainty":
            # Samples for which DGO was most uncertain (if uncertainty metric available)
            uncertain_samples = [s for s in parent_candidates_meta if s.dgo_uncertainty is not None]
            if uncertain_samples:
                uncertain_samples.sort(key=lambda s: s.dgo_uncertainty,
                                       reverse=True)  # Higher uncertainty is "better" parent
                selected_meta = random.choice(uncertain_samples[:max(1, len(uncertain_samples) // 5)])
            else:  # Fallback
                selected_meta = random.choice(parent_candidates_meta)

        else:  # Default/Unknown strategy
            logger.warning(f"Unknown parent selection strategy: {strategy}. Using random.")
            selected_meta = random.choice(parent_candidates_meta)

        if selected_meta is None:  # Should ideally not happen if parent_candidates_meta is not empty
            logger.error("Parent selection failed to choose a meta sample. Fallback to initial seed.")
            return random.choice(self.initial_seed_samples).copy()

        # Load the image for the selected parent
        parent_image = self.morph_library.get_sample_image(selected_meta.sample_id)
        if parent_image is None:
            logger.error(f"Failed to load parent image for ID {selected_meta.sample_id}. Fallback to initial seed.")
            return random.choice(self.initial_seed_samples).copy()

        logger.debug(
            f"Selected parent ID: {selected_meta.sample_id} using strategy '{strategy}'. Confidence: {selected_meta.dgo_confidence:.3f}")
        return parent_image.copy()

    def _trigger_dgo_finetuning_if_needed(self):
        """Checks if DGO finetuning should be triggered and performs it."""
        if self.morph_library.get_count() >= self.last_dgo_finetune_library_size + self.gen_cfg.dgo_finetune_trigger_new_samples \
                and len(self.dgo_finetune_data_buffer) >= getattr(self.gen_cfg, 'min_samples_for_dgo_finetune',
                                                                  20):  # Add to GenCfg

            logger.info("Triggering DGO fine-tuning...")
            # Get training engine instance (could be member if stateful like SI)
            from ..dgo_oracle.training_engine import DGOTrainingEngine  # Local import
            trainer = DGOTrainingEngine(self.dgo_handler, self.cfg.dgo_oracle.training_params,
                                        self.cfg.dgo_oracle, self.cfg.data_management, self.device)

            # Prepare data for finetuning from the buffer
            # Data in buffer is (CvImage, label)
            # DGOTrainingEngine's finetune_model currently expects List[(ImagePath, Label)] due to create_dgo_dataloader
            # This needs to be reconciled. For now, assume an adapter or that finetune_model can handle CvImage.

            # Adapter: Save CvImages to temp files for finetuning_model
            # This is inefficient but matches current DGOTrainingEngine expectation.
            # A better DGOTrainingEngine would accept (Tensor, Label) or (CvImage, Label) directly.
            temp_paths_for_finetune: List[Tuple[ImagePath, int]] = []
            temp_finetune_dir = "./temp_dgo_finetune_imgs/"
            os.makedirs(temp_finetune_dir, exist_ok=True)

            # Use a representative sample from the buffer
            finetune_samples_to_use = list(self.dgo_finetune_data_buffer)  # Take all or a random subset
            if len(finetune_samples_to_use) > self.gen_cfg.dgo_finetune_data_buffer_size * 0.8:  # Heuristic: use recent subset
                finetune_samples_to_use = random.sample(finetune_samples_to_use,
                                                        int(self.gen_cfg.dgo_finetune_data_buffer_size * 0.8))

            for i, (cv_img, label) in enumerate(finetune_samples_to_use):
                try:
                    temp_path_str = os.path.join(temp_finetune_dir, f"finetune_img_{i}.png")
                    cv2.imwrite(temp_path_str, cv_img)
                    temp_paths_for_finetune.append((ImagePath(temp_path_str), label))
                except Exception as e_write:
                    logger.error(f"Error writing temp image for DGO finetuning: {e_write}")

            if temp_paths_for_finetune:
                trainer.finetune_model(temp_paths_for_finetune)  # Pass List[(ImagePath, Label)]

                # After finetuning, feature space might have changed.
                # Update quality assessor's view of library features and re-fit KDE.
                logger.info("DGO finetuned. Re-evaluating library features for QualityAssessor.")
                self.quality_assessor.update_library_state(
                    self.morph_library.get_all_hashes(),
                    self._get_all_library_features()  # Re-fetch all features
                )
                # Also, update DGOModelHandler's knowledge of its feature dim if it changed (unlikely for finetune)
                new_feat_dim = self.dgo_handler.get_feature_dimensionality()
                if new_feat_dim and new_feat_dim != self.feature_dim:
                    logger.warning(
                        f"DGO feature dimensionality changed from {self.feature_dim} to {new_feat_dim} after finetuning!")
                    self.feature_dim = new_feat_dim
                    # Re-initialize Hasher and Analyzer if feature_dim changes. This is major.
                    self.feature_hasher = AdvancedFeatureHasher(self.feature_dim, self.cfg.feature_analysis,
                                                                self.device)
                    if self.library_analyzer:
                        self.library_analyzer = LibraryDiversityAnalyzer(self.feature_dim, self.cfg.feature_analysis)
                    # QualityAssessor also needs new hasher/analyzer instances
                    self.quality_assessor.feature_hasher = self.feature_hasher
                    self.quality_assessor.library_analyzer = self.library_analyzer
                    # This would also invalidate all existing hashes in the library! Re-hashing needed.
                    logger.error(
                        "Feature dimensionality change requires re-hashing entire library. Not implemented yet.")

            # Clean up temp images
            for p_tpl in temp_paths_for_finetune:
                if os.path.exists(p_tpl[0]): os.remove(p_tpl[0])
            if os.path.exists(temp_finetune_dir):
                try:
                    os.rmdir(temp_finetune_dir)
                except OSError:
                    pass  # If not empty due to error, leave it

            self.last_dgo_finetune_library_size = self.morph_library.get_count()
            self.dgo_finetune_data_buffer.clear()  # Clear buffer after use, or use a better sampling strategy

    def _check_stopping_conditions(self) -> bool:
        """Checks if generation should stop based on configured conditions."""
        if self.morph_library.get_count() >= self.gen_cfg.max_library_size:
            logger.info(f"Stopping: Max library size ({self.gen_cfg.max_library_size}) reached.")
            return True
        if self.consecutive_no_new_samples >= self.gen_cfg.max_generations_without_improvement:
            logger.info(
                f"Stopping: Max generations without new sample ({self.gen_cfg.max_generations_without_improvement}) reached.")
            return True

        # Add other conditions: DGO performance on a validation set, time limit, etc.
        # from .stopping_conditions import check_dgo_validation_performance # If implemented
        # if check_dgo_validation_performance(self.dgo_handler, ...): return True

        return False

    def run_generation_loop(self):
        """Main iterative loop for generating character morphologies."""
        logger.info(f"Starting generation loop for character '{self.target_char_string}'.")
        logger.info(
            f"Max library size: {self.gen_cfg.max_library_size}, Max no improvement streak: {self.gen_cfg.max_generations_without_improvement}")

        if not self.initial_seed_samples:  # Final check
            logger.critical("Cannot start generation loop: No initial seed samples.")
            return

        while not self._check_stopping_conditions():
            self.generation_step += 1
            logger.info(
                f"--- Generation Step {self.generation_step} --- Lib Size: {self.morph_library.get_count()} ---")

            parent_image = self._select_parent_sample()
            # cv2.imshow("Selected Parent", parent_image); cv2.waitKey(1) # Debug

            # Perturb (can be single or sequence, or DGO-guided)
            # For now, apply a short sequence of random basic perturbations.
            # More advanced: orchestrator decides based on state (e.g., use DGO-guided if exploring boundary)

            num_perturb_steps = random.randint(1, getattr(self.cfg.perturbation_suite, 'max_perturb_sequence_len',
                                                          3))  # Add to PertSuiteCfg

            # Decide if to use DGO-guided perturbation
            use_dgo_guided_prob = getattr(self.cfg.perturbation_suite, 'dgo_guided_perturb_probability',
                                          0.1)  # Add to PertSuiteCfg

            candidate_image: CvImage
            perturb_name_str: str
            perturb_params_dict: Dict

            if self.dgo_handler and random.random() < use_dgo_guided_prob:
                # DGO-guided perturbation
                # Could try to make it less like current target (to find boundary) or more like a confusable char
                # Or more like target if confidence is low but structure is OK.
                # Simple strategy: try to reduce confidence in target if it's already high, or increase if low.
                # This needs DGO output of parent first.
                parent_dgo_pred_idx, parent_conf, _, _, _ = self.dgo_handler.predict(parent_image)

                guide_strength = random.uniform(0.005, 0.03)  # Small epsilon for FGSM-like
                misguide_target = None
                if parent_dgo_pred_idx == self.target_char_idx and parent_conf > 0.7:
                    # If confident target, try to make it *less* like target (explore boundary)
                    # This means gradient of logit[target_idx] should be used to *decrease* it.
                    # Or, if misguiding, pick a confusable char.
                    if self.cfg.confusable_character_indices and random.random() < 0.5:
                        misguide_target = random.choice(self.cfg.confusable_character_indices)
                    # How get_gradient_wrt_input defines loss for target_class_idx matters.
                    # If loss=logit(target), then to decrease it, step *against* gradient.
                    # Or, if loss=-logit(target), step *along* gradient.
                    # The current get_gradient_wrt_input with loss=logit[target_idx] returns grad that *increases* logit.
                    # So, for FGSM-like to reduce confidence, use -strength * sign(grad).
                    # Or, if using raw grad, -strength * grad.
                    # For simplicity, let current DGO-guided method handle directionality based on misguide_target.
                    # Here, we want to explore, so either misguide or make less like target.
                    logger.debug(
                        f"Applying DGO-guided perturbation (explore boundary from high conf parent). Misguide to: {misguide_target}")

                candidate_image, perturb_name_str, perturb_params_dict = \
                    self.perturb_orchestrator.apply_dgo_guided_perturbation(
                        parent_image.copy(), target_char_idx=self.target_char_idx,
                        misguide_to_idx=misguide_target, strength=guide_strength
                    )
            else:
                # Standard random perturbation sequence
                candidate_image, p_info_list = self.perturb_orchestrator.apply_perturbation_sequence(
                    parent_image.copy(), num_perturbations=num_perturb_steps, enforce_distinct_methods=True
                )
                if p_info_list:
                    # For logging, take first applied perturbation in sequence, or summarize
                    perturb_name_str = "; ".join(sorted(list(set(p[0] for p in p_info_list))))  # Unique method names
                    perturb_params_dict = {"sequence_length": len(p_info_list),
                                           "first_params": p_info_list[0][1] if p_info_list else {}}
                else:
                    perturb_name_str = "none_applied_in_sequence"
                    perturb_params_dict = {}

            # cv2.imshow("Candidate", candidate_image); cv2.waitKey(1) # Debug

            # Assess candidate quality
            # (DGO prediction is done inside assessor if not precomputed)
            assessment_results = self.quality_assessor.assess_candidate(candidate_image)

            logger.info(f"Candidate assessment: Target? {assessment_results['is_target_char']}, "
                        f"DGO Conf: {assessment_results['dgo_confidence']:.3f}, "
                        f"Structure? {assessment_results['passes_structure_check']}, "
                        f"Novel? {assessment_results['is_novel']} (Score: {assessment_results['novelty_score']:.1f}), "
                        f"Reason: {assessment_results['acceptance_reason']}")

            # Add to DGO finetuning buffer (even if not added to library, can be useful negative/hard sample)
            # Label for finetuning: DGO's prediction if fairly confident, or true target if it *should* be target.
            # This needs careful thought. For now, use DGO's prediction if not wildly off.
            dgo_pred_for_buffer = assessment_results['dgo_predicted_label']
            # If it passed structure and DGO says it's target (even if low conf), maybe label as target.
            if assessment_results['is_target_char'] and assessment_results['passes_structure_check']:
                dgo_label_for_buffer = self.target_char_idx
            else:  # Otherwise, trust DGO's label, or use a "don't know/other" label.
                dgo_label_for_buffer = assessment_results['dgo_predicted_label']

            self.dgo_finetune_data_buffer.append((candidate_image.copy(), dgo_label_for_buffer))

            if assessment_results['should_be_added_to_library']:
                parent_id_for_db = None  # TODO: Need to get ID of parent if parent was from library

                # Add to morphology library (DB and file storage)
                added_sample_obj = self.morph_library.add_new_sample(
                    image_data=candidate_image,  # This will be standardized again by MorphologyLibrary
                    image_hash=np.array(assessment_results['image_hash']) if assessment_results[
                        'image_hash'] else np.array([]),
                    dgo_pred_label=assessment_results['dgo_predicted_label'],
                    dgo_confidence=assessment_results['dgo_confidence'],
                    generation_step=self.generation_step,
                    structure_passed=assessment_results['passes_structure_check'],
                    feature_vector=assessment_results['feature_vector'],
                    dgo_uncertainty=assessment_results['dgo_uncertainty'],
                    parent_id=parent_id_for_db,  # This needs to be tracked
                    perturb_name=perturb_name_str,
                    perturb_params=perturb_params_dict,
                    novelty_score=assessment_results['novelty_score']
                )
                if added_sample_obj:
                    self.consecutive_no_new_samples = 0
                    # Update quality assessor's view of the library
                    self.quality_assessor.update_library_state(
                        self.morph_library.get_all_hashes(),  # Re-fetch all
                        self._get_all_library_features()  # Re-fetch all
                    )
                else:
                    logger.warning("Candidate assessed as addable, but failed to save to library.")
                    self.consecutive_no_new_samples += 1
            else:
                self.consecutive_no_new_samples += 1

            # Trigger DGO finetuning if conditions met
            self._trigger_dgo_finetuning_if_needed()

            # Adaptive updates to orchestrator (placeholder)
            # self.perturb_orchestrator.update_method_weights(feedback_from_assessment)
            # self.perturb_orchestrator.adapt_perturbation_parameters(method_applied, feedback)

        logger.info("Generation loop finished.")
        # cv2.destroyAllWindows() # If using imshow


if __name__ == "__main__":
    # --- Test GenerationManager ---
    # This is a more complex integration test. Requires careful config.
    # For a simple test run, many features might be disabled or use defaults.

    # Use a minimal config for testing the loop flow
    from ..config import SystemConfig  # Adjust relative import

    # Path for dummy initial 'X' sample
    dummy_initial_X_dir = "./temp_initial_gm_test/char_X/"
    os.makedirs(dummy_initial_X_dir, exist_ok=True)
    img_X_ref = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(img_X_ref, "X", (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)
    cv2.imwrite(os.path.join(dummy_initial_X_dir, "initial_X_0.png"), img_X_ref)

    test_sys_cfg_data_gm = {
        "project_name": "GenManagerTest",
        "target_character_idx": 0,  # Assuming 'X' is class 0 for dummy DGO
        "target_character_string": "X",
        "random_seed": 42,

        "data_management": {
            "initial_samples_path_template": "./temp_initial_gm_test/char_{char_string}/",
            "output_base_dir": "./temp_generated_gm_output/",
            "target_image_size": (28, 28), "grayscale_input": True
        },
        "dgo_oracle": {
            "model_architecture": "BaseCNN", "num_classes": 2,  # Binary: X vs Not-X, or just few classes
            "pretrained_model_path": None,
            "feature_extraction_layer_name": "relu3",
            "training_params": {"epochs_initial_training": 0, "epochs_finetuning": 0, "batch_size": 2},
            # No actual training in test
            "continual_learning_strategy": "none"
        },
        "perturbation_suite": {
            "local_pixel": {"enabled": True, "probability_of_application": 1.0},  # Ensure one method is always active
            # Disable others for simplicity of test run
            "elastic_deformation": {"enabled": False}, "fine_affine": {"enabled": False},
            "stroke_thickness_morph": {"enabled": False},
            "stroke_engine_perturbations": {"enabled": False},
            "style_mixer": {"enabled": False},
            "max_perturb_sequence_len": 1,  # Add this to PerturbationSuiteConfig model
            "dgo_guided_perturb_probability": 0.0  # Disable DGO-guided for basic test
        },
        "structure_guard": {
            "basic_topology": {"enabled": True, "rules_for_char": {"X": {"expected_connected_components": 1}}},
            # Basic rule for X
            "advanced_topology": {"enabled": False}
        },
        "feature_analysis": {
            "hashing_method": "simhash_projection", "hash_length": 16,
            "novelty_hamming_distance_threshold_ratio": 0.2,
            "diversity_strategy": "novelty_only"  # Simplest diversity
        },
        "generation_manager": {
            "max_library_size": 5,  # Run for very few samples
            "max_generations_without_improvement": 3,
            "dgo_acceptance_confidence_threshold": 0.1,  # Low threshold for test
            "dgo_finetune_trigger_new_samples": 100,  # Effectively disable finetuning
            "parent_selection_strategy": "random",
            "num_initial_seeds": 1  # Add to GenManagerConfig
        },
        "logging": {"level": "INFO"}  # Use INFO to reduce verbosity for test run
    }
    # Add missing fields to Pydantic models if test reveals them
    # E.g., PerturbationSuiteConfig might need max_perturb_sequence_len
    # GenerationManagerConfig might need num_initial_seeds, min_library_size_for_advanced_parent_selection etc.
    # Ensure these are added to the actual Pydantic models in config.py.

    # For this test, quickly add them to the dict if not in model, knowing Pydantic will ignore extra fields
    if 'max_perturb_sequence_len' not in PerturbationSuiteConfig.model_fields:
        test_sys_cfg_data_gm["perturbation_suite"]['max_perturb_sequence_len'] = 1
    if 'num_initial_seeds' not in GenerationManagerConfig.model_fields:
        test_sys_cfg_data_gm["generation_manager"]['num_initial_seeds'] = 1

    from ..config import _config_instance  # Adjust relative import

    _config_instance = SystemConfig.model_validate(test_sys_cfg_data_gm)

    # Initialize and run
    if _config_instance.generation_manager.num_initial_seeds == 0 and not _config_instance.data_management.initial_samples_path_template:
        print("Test config error: num_initial_seeds is 0 and no initial_samples_path_template. Test will fail.")
    else:
        print("\n--- Testing GenerationManager Loop (Minimal Run) ---")
        try:
            manager = GenerationManager(cfg_system=_config_instance)  # Pass the validated config
            manager.run_generation_loop()
            print(f"GenerationManager loop test finished. Library items: {manager.morph_library.get_count()}")
            assert manager.morph_library.get_count() <= manager.gen_cfg.max_library_size
        except ValueError as ve:  # Catch potential "No initial seed samples"
            print(f"ValueError during GenerationManager init or run: {ve}")
        except Exception as e:
            print(f"Exception during GenerationManager test: {e}", exc_info=True)

    # Clean up temporary directories
    import shutil

    shutil.rmtree("./temp_initial_gm_test/", ignore_errors=True)
    shutil.rmtree("./temp_generated_gm_output/", ignore_errors=True)
    shutil.rmtree("./temp_dgo_finetune_imgs/", ignore_errors=True)  # If created

    print("\nGenerationManager tests completed.")