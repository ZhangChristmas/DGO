# ultimate_morph_generator/core_logic/quality_assessor.py
import numpy as np
from typing import Dict, Any, Optional, List

from ..config import get_config, GenerationManagerConfig  # For thresholds etc.
from ..utilities.type_definitions import CvImage, FeatureVector, DGOOutput, MorphologySample
from ..dgo_oracle.dgo_model_handler import DGOModelHandler  # For DGO scores
from ..feature_analysis.advanced_hashing import AdvancedFeatureHasher  # For novelty/distance
from ..feature_analysis.diversity_metrics import LibraryDiversityAnalyzer  # For density
from ..structure_guard import StructureGuard  # For structural integrity
from ..utilities.logging_config import setup_logging

logger = setup_logging()


class CandidateQualityAssessor:
    """
    Assesses the overall quality of a generated candidate image based on multiple criteria:
    - DGO recognition (identity, confidence, uncertainty).
    - Structural integrity (from StructureGuard).
    - Novelty/Diversity (compared to existing library, using feature hashes and/or feature space density).
    - (Optionally) Visual appeal or specific adversarial properties (harder to quantify).
    """

    def __init__(self,
                 target_char_idx: int,
                 gen_cfg: GenerationManagerConfig,
                 dgo_handler: DGOModelHandler,
                 structure_checker: StructureGuard,
                 feature_hasher: AdvancedFeatureHasher,  # Used for novelty calculation
                 library_analyzer: Optional[LibraryDiversityAnalyzer] = None,  # For density-based scores
                 morphology_library_hashes: Optional[List[np.ndarray]] = None,  # Current hashes in library
                 morphology_library_features: Optional[List[FeatureVector]] = None  # Current features for density
                 ):
        self.target_char_idx = target_char_idx
        self.gen_cfg = gen_cfg  # Contains thresholds like DGO_CONFIDENCE_LOWER_BOUND

        self.dgo_handler = dgo_handler
        self.structure_checker = structure_checker
        self.feature_hasher = feature_hasher
        self.library_analyzer = library_analyzer

        # These are dynamic and should be updated as the library grows
        self.current_library_hashes = morphology_library_hashes if morphology_library_hashes else []
        self.current_library_features = morphology_library_features if morphology_library_features else []

    def update_library_state(self,
                             library_hashes: List[np.ndarray],
                             library_features: List[FeatureVector]):
        """Call this when the main library state changes."""
        self.current_library_hashes = library_hashes
        self.current_library_features = library_features
        # Re-fit KDE if used by library_analyzer (or analyzer handles this internally)
        if self.library_analyzer and self.library_analyzer.kde_estimator and self.current_library_features:
            try:
                feature_matrix = np.array(self.current_library_features)
                if feature_matrix.ndim == 2 and feature_matrix.shape[0] > 0:
                    self.library_analyzer.kde_estimator.fit(feature_matrix)
                    logger.debug("QualityAssessor: KDE estimator re-fitted with updated library features.")
            except Exception as e:
                logger.error(f"Error re-fitting KDE in QualityAssessor: {e}")

    def assess_candidate(self,
                         candidate_image: CvImage,
                         # Pass DGO output if already computed to avoid re-computation
                         dgo_output_precomputed: Optional[DGOOutput] = None
                         ) -> Dict[str, Any]:
        """
        Evaluates a candidate image and returns a dictionary of quality scores/flags.
        Keys in output dict:
            'is_target_char': bool
            'dgo_confidence': float
            'dgo_predicted_label': int
            'dgo_uncertainty': Optional[float]
            'passes_structure_check': bool
            'is_novel': bool
            'novelty_score': float (e.g., min Hamming distance or 1 - max_similarity)
            'feature_vector': Optional[FeatureVector]
            'image_hash': Optional[ImageHash]
            'feature_space_density': Optional[float] (lower is better for exploration)
            'overall_acceptability_score': float (a combined score, if desired)
            'acceptance_reason': str (e.g. "Meets all criteria", "Edge case", "Adversarial but correct")
            'should_be_added_to_library': bool
        """
        assessment = {
            "should_be_added_to_library": False,
            "acceptance_reason": "Not assessed"
        }

        # 1. DGO Evaluation (Identity, Confidence, Features, Uncertainty)
        if dgo_output_precomputed:
            pred_idx, confidence, _, features_np, uncertainty = dgo_output_precomputed
            assessment['dgo_raw_output_provided'] = True
        else:
            pred_idx, confidence, _, features_np, uncertainty = self.dgo_handler.predict(candidate_image)
            assessment['dgo_raw_output_provided'] = False

        assessment['dgo_predicted_label'] = pred_idx
        assessment['dgo_confidence'] = float(confidence)
        assessment['feature_vector'] = features_np  # Can be None if extraction failed
        assessment['dgo_uncertainty'] = uncertainty

        assessment['is_target_char'] = (pred_idx == self.target_char_idx)

        dgo_pass = assessment['is_target_char'] and \
                   assessment['dgo_confidence'] >= self.gen_cfg.dgo_acceptance_confidence_threshold

        if not dgo_pass:
            assessment[
                'acceptance_reason'] = f"DGO check failed (Target: {assessment['is_target_char']}, Conf: {confidence:.3f})"
            # No need to proceed further if DGO rejects it for basic acceptance
            # But for analysis, we might still want other scores. Let's continue for now.
            # return assessment # Early exit if strict DGO pass is first gate

        # 2. Structural Integrity Check
        # Note: StructureGuard might need pre-computed strokes. For now, assume it handles image input.
        passes_structure = self.structure_checker.check_image_structure(candidate_image.copy())
        assessment['passes_structure_check'] = passes_structure
        if not passes_structure:
            assessment['acceptance_reason'] = "Structure check failed."
            # return assessment # Early exit

        # 3. Novelty Check (using Feature Hashes)
        novelty_score = 0.0  # Higher is more novel
        is_novel = True  # Assume novel if no library or no features
        image_hash_np: Optional[np.ndarray] = None

        if features_np is not None and features_np.size > 0:
            image_hash_np = self.feature_hasher.get_hash(features_np)
            assessment['image_hash'] = image_hash_np.tolist() if image_hash_np is not None else None  # For dict

            if self.current_library_hashes and len(self.current_library_hashes) > 0:
                min_dist = self.feature_hasher.hash_length + 1
                for existing_hash in self.current_library_hashes:
                    dist = AdvancedFeatureHasher.hamming_distance(image_hash_np, existing_hash)
                    min_dist = min(min_dist, dist)

                novelty_threshold_abs = int(self.feature_hasher.hash_length * \
                                            get_config().feature_analysis.novelty_hamming_distance_threshold_ratio)

                if min_dist < novelty_threshold_abs:
                    is_novel = False
                novelty_score = float(min_dist)  # Raw min distance as score
            else:  # Library is empty, first item is always novel
                is_novel = True
                novelty_score = float(self.feature_hasher.hash_length)  # Max possible distance
        else:  # No features, cannot compute hash or novelty based on features
            is_novel = False  # Or True if we want to accept images without features? Risky.
            assessment['image_hash'] = None
            logger.debug("No features extracted, cannot compute hash-based novelty.")

        assessment['is_novel'] = is_novel
        assessment['novelty_score'] = novelty_score

        if not is_novel and dgo_pass and passes_structure:  # Update reason if this is the failing point
            assessment['acceptance_reason'] = "Not novel enough."
            # return assessment # Early exit

        # 4. Feature Space Density (Optional, if library_analyzer is provided)
        density_score: Optional[float] = None
        if self.library_analyzer and features_np is not None and features_np.size > 0 and \
                len(self.current_library_features) > 0:  # Need existing features to compare against
            try:
                # Density is P(x), lower is "better" for exploration.
                # score_samples returns log-density.
                densities_at_query = self.library_analyzer.estimate_feature_space_density_kde(
                    feature_vectors=self.current_library_features,  # Fit on current library
                    query_points=[FeatureVector(features_np)]  # Score the candidate
                )
                if densities_at_query is not None and len(densities_at_query) > 0:
                    density_score = float(densities_at_query[0])
                    # We might want to normalize this or use log-density.
                    # For now, raw density.
            except Exception as e:
                logger.error(f"Error getting density score: {e}")
        assessment['feature_space_density'] = density_score

        # 5. Final Decision Logic (should_be_added_to_library and reason)
        # This is where different acceptance criteria can be implemented.
        # Example: Must pass DGO, structure, AND novelty.
        if dgo_pass and passes_structure and is_novel:
            assessment['should_be_added_to_library'] = True
            reason_parts = [f"DGO Conf: {confidence:.2f}"]
            if assessment['dgo_uncertainty'] is not None:
                reason_parts.append(f"Uncert: {assessment['dgo_uncertainty']:.2f}")
            reason_parts.append(f"Novelty: {novelty_score:.0f}")
            if density_score is not None:
                reason_parts.append(f"Density: {density_score:.2e}")

            # Categorize acceptance:
            if confidence < self.gen_cfg.dgo_boundary_case_confidence_range[1] and \
                    confidence > self.gen_cfg.dgo_boundary_case_confidence_range[0]:
                assessment['acceptance_category'] = "BoundaryCase"
            elif confidence >= self.gen_cfg.dgo_high_confidence_threshold:
                assessment['acceptance_category'] = "HighConfidence"
            else:  # Standard acceptance
                assessment['acceptance_category'] = "Standard"

            assessment['acceptance_reason'] = f"Accepted ({assessment['acceptance_category']}): " + ", ".join(
                reason_parts)

        elif dgo_pass and passes_structure and not is_novel:
            assessment['acceptance_reason'] = "Rejected: DGO/Structure OK, but not novel."
        elif dgo_pass and not passes_structure:
            assessment['acceptance_reason'] = "Rejected: DGO OK, but structure failed."
        elif not dgo_pass:  # Already handled but for completeness
            assessment[
                'acceptance_reason'] = f"Rejected: DGO check failed (Target: {assessment['is_target_char']}, Conf: {confidence:.3f})"
        else:  # Other combination
            assessment['acceptance_reason'] = "Rejected: Multiple criteria failed."

        # (Optional) Calculate an overall score if needed for ranking candidates
        # score_weights = {"conf": 0.4, "novelty": 0.3, "structure": 0.3} # Example
        # overall_score = (float(dgo_pass) * score_weights["structure"] + \
        #                  confidence * score_weights["conf"] + \
        #                  (novelty_score / self.feature_hasher.hash_length) * score_weights["novelty"])
        # assessment['overall_acceptability_score'] = overall_score

        return assessment


if __name__ == "__main__":
    # --- Test CandidateQualityAssessor ---
    from ..config import SystemConfig  # Adjust relative import
    from ..dgo_oracle.dgo_model_handler import DGOModelHandler  # For dummy DGO
    from ..structure_guard import StructureGuard, compute_reference_ph_signature_for_char  # For dummy SG
    from ..feature_analysis.advanced_hashing import AdvancedFeatureHasher
    from ..feature_analysis.diversity_metrics import LibraryDiversityAnalyzer

    # Create dummy config for testing
    temp_sys_cfg_data_qa = {
        "target_character_idx": 3,
        "target_character_string": "3",  # For structure guard
        "generation_manager": {  # GenManagerConfig for thresholds
            "dgo_acceptance_confidence_threshold": 0.5,
            "dgo_boundary_case_confidence_range": (0.4, 0.7),
            "dgo_high_confidence_threshold": 0.9
        },
        "dgo_oracle": {  # For dummy DGOHandler
            "model_architecture": "BaseCNN", "num_classes": 10,
            "feature_extraction_layer_name": "relu3"
        },
        "data_management": {"target_image_size": (28, 28), "grayscale_input": True},
        "structure_guard": {  # For dummy StructureGuard
            "basic_topology": {"enabled": True, "rules_for_char": {"3": {"expected_holes": 0}}},
            "advanced_topology": {"enabled": False}  # Keep advanced off for simple test
        },
        "feature_analysis": {  # For Hasher and Analyzer
            "hashing_method": "simhash_projection", "hash_length": 32,
            "novelty_hamming_distance_threshold_ratio": 0.1,
            "diversity_strategy": "explore_low_density", "feature_space_density_estimator": "kde",
            "kde_bandwidth": 0.2
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_qa)
    cfg_glob_qa = get_config()

    # Dummy components
    dummy_dgo = DGOModelHandler(cfg_glob_qa.dgo_oracle, cfg_glob_qa.data_management, cfg_glob_qa.get_actual_device())


    # Mock DGO predict to control its output for tests
    def mock_dgo_predict_high_conf(img_input):
        feat_dim = dummy_dgo.get_feature_dimensionality() or 32
        return (cfg_glob_qa.target_character_idx, 0.95, np.random.rand(10),
                np.random.rand(feat_dim).astype(np.float32), 0.05)


    def mock_dgo_predict_low_conf_correct_label(img_input):
        feat_dim = dummy_dgo.get_feature_dimensionality() or 32
        return (cfg_glob_qa.target_character_idx, 0.3, np.random.rand(10),
                np.random.rand(feat_dim).astype(np.float32), 0.2)


    def mock_dgo_predict_wrong_label(img_input):
        feat_dim = dummy_dgo.get_feature_dimensionality() or 32
        wrong_idx = (cfg_glob_qa.target_character_idx + 1) % 10
        return (wrong_idx, 0.8, np.random.rand(10),
                np.random.rand(feat_dim).astype(np.float32), 0.1)


    dummy_sg = StructureGuard(cfg_glob_qa.structure_guard, cfg_glob_qa.target_character_string)


    # Mock SG check_image_structure
    def mock_sg_pass(img, strokes=None):
        return True


    def mock_sg_fail(img, strokes=None):
        return False


    feat_dim_actual = dummy_dgo.get_feature_dimensionality() or 32  # Use actual inferred dim
    dummy_hasher = AdvancedFeatureHasher(feat_dim_actual, cfg_glob_qa.feature_analysis)
    dummy_analyzer = LibraryDiversityAnalyzer(feat_dim_actual, cfg_glob_qa.feature_analysis)

    # Test image
    test_candidate_img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)

    # Initial library state (empty)
    lib_hashes = []
    lib_features = []

    assessor = CandidateQualityAssessor(
        target_char_idx=cfg_glob_qa.target_character_idx,
        gen_cfg=cfg_glob_qa.generation_manager,
        dgo_handler=dummy_dgo,
        structure_checker=dummy_sg,
        feature_hasher=dummy_hasher,
        library_analyzer=dummy_analyzer,
        morphology_library_hashes=lib_hashes,
        morphology_library_features=lib_features
    )

    print("\n--- Testing Quality Assessor ---")

    # Scenario 1: Good candidate (high conf, structure pass, novel)
    print("\nScenario 1: Good Candidate")
    dummy_dgo.predict = mock_dgo_predict_high_conf  # Override with mock
    dummy_sg.check_image_structure = mock_sg_pass
    assessment1 = assessor.assess_candidate(test_candidate_img)
    print(f"Assessment 1: {assessment1}")
    assert assessment1['should_be_added_to_library']
    assert assessment1['is_target_char'] and assessment1['passes_structure_check'] and assessment1['is_novel']

    # Update library for next assessment
    if assessment1['should_be_added_to_library'] and assessment1['image_hash'] is not None and assessment1[
        'feature_vector'] is not None:
        lib_hashes.append(np.array(assessment1['image_hash']))
        lib_features.append(assessment1['feature_vector'])
        assessor.update_library_state(lib_hashes, lib_features)  # Important!

    # Scenario 2: Not novel (using features/hash from assessment1)
    print("\nScenario 2: Not Novel")
    # DGO and structure still good, but use same features as assessment1
    dgo_out_for_scen2 = (
        cfg_glob_qa.target_character_idx, 0.96, np.random.rand(10),
        assessment1['feature_vector'],  # Use same feature vector
        0.04
    )
    dummy_sg.check_image_structure = mock_sg_pass  # Structure still passes
    assessment2 = assessor.assess_candidate(test_candidate_img, dgo_output_precomputed=dgo_out_for_scen2)
    print(f"Assessment 2: {assessment2}")
    assert not assessment2['should_be_added_to_library']
    assert not assessment2['is_novel']
    assert assessment2['passes_structure_check'] and assessment2['is_target_char']

    # Scenario 3: Fails DGO (low confidence)
    print("\nScenario 3: Fails DGO (Low Confidence)")
    dummy_dgo.predict = mock_dgo_predict_low_conf_correct_label
    dummy_sg.check_image_structure = mock_sg_pass
    assessment3 = assessor.assess_candidate(test_candidate_img.copy())  # New image copy
    print(f"Assessment 3: {assessment3}")
    assert not assessment3['should_be_added_to_library']
    assert not (assessment3['is_target_char'] and assessment3[
        'dgo_confidence'] >= cfg_glob_qa.generation_manager.dgo_acceptance_confidence_threshold)

    # Scenario 4: Fails Structure Check
    print("\nScenario 4: Fails Structure Check")
    dummy_dgo.predict = mock_dgo_predict_high_conf
    dummy_sg.check_image_structure = mock_sg_fail  # Structure now fails
    assessment4 = assessor.assess_candidate(test_candidate_img.copy())
    print(f"Assessment 4: {assessment4}")
    assert not assessment4['should_be_added_to_library']
    assert not assessment4['passes_structure_check']

    print("\nCandidateQualityAssessor tests completed.")