# ultimate_morph_generator/structure_guard/__init__.py
from typing import Optional, List

from ..config import get_config, StructureGuardConfig
from ..utilities.type_definitions import CvImage
from ..perturbation_suite.stroke_engine.stroke_extractor import Stroke  # If passing strokes
from ..utilities.logging_config import setup_logging

from .basic_topology import BasicTopologyChecker
from .advanced_topology.persistent_homology import PersistentHomologyAnalyzer, PersistenceSignature
from .advanced_topology.graph_representation import CharacterGraphAnalyzer

logger = setup_logging()


class StructureGuard:
    """
    Orchestrates various structural and topological checks to ensure
    perturbed characters maintain their fundamental form.
    """

    def __init__(self, guard_cfg: StructureGuardConfig,
                 target_char_string: str,
                 reference_ph_signature: Optional[PersistenceSignature] = None,  # Pre-computed for target char
                 # reference_char_graph: Optional[Any] = None # Pre-computed/defined NetworkX graph
                 ):
        self.cfg = guard_cfg
        self.target_char_string = target_char_string

        self.basic_checker: Optional[BasicTopologyChecker] = None
        if self.cfg.basic_topology.enabled:
            self.basic_checker = BasicTopologyChecker(
                topology_cfg=self.cfg.basic_topology,
                target_char_string=self.target_char_string
            )
            logger.info("BasicTopologyChecker initialized.")

        self.ph_analyzer: Optional[PersistentHomologyAnalyzer] = None
        if self.cfg.advanced_topology.enabled and self.cfg.advanced_topology.persistent_homology_params:
            # Check if persistent_homology_params itself is enabled, not just advanced_topology
            # This is implied if adv_topo_cfg.enabled is true and ph_params exists.
            # A specific enable flag within persistent_homology_params is better.
            # For now, assume if advanced_topology is enabled, its sub-modules are too if configured.
            ph_specific_enabled = self.cfg.advanced_topology.persistent_homology_params.get("enabled", True)
            if ph_specific_enabled:
                self.ph_analyzer = PersistentHomologyAnalyzer(
                    adv_topology_cfg=self.cfg.advanced_topology
                )
                self.reference_ph_signature = reference_ph_signature
                if self.reference_ph_signature:
                    logger.info("PersistentHomologyAnalyzer initialized with reference signature.")
                else:
                    logger.warning(
                        "PersistentHomologyAnalyzer initialized, but NO reference PH signature provided. PH checks might be limited to raw signature generation.")
            else:
                logger.info("Persistent Homology sub-module explicitly disabled in config.")

        self.graph_analyzer: Optional[CharacterGraphAnalyzer] = None
        if self.cfg.advanced_topology.enabled and self.cfg.advanced_topology.graph_representation_params:
            graph_specific_enabled = self.cfg.advanced_topology.graph_representation_params.get("enabled", True)
            if graph_specific_enabled:
                self.graph_analyzer = CharacterGraphAnalyzer(
                    adv_topology_cfg=self.cfg.advanced_topology,
                    target_char_string=self.target_char_string
                )
                # self.reference_char_graph = reference_char_graph (passed in or loaded by analyzer)
                if self.graph_analyzer.reference_graph:  # Analyzer loads its own ref graph
                    logger.info("CharacterGraphAnalyzer initialized with reference graph.")
                else:
                    logger.warning(
                        "CharacterGraphAnalyzer initialized, but NO reference graph available. Graph checks will fail or be skipped.")
            else:
                logger.info("Graph Representation sub-module explicitly disabled in config.")

    def check_image_structure(self, image: CvImage,
                              extracted_strokes: Optional[List[Stroke]] = None) -> bool:
        """
        Performs all enabled structural checks on the image.
        Args:
            image: The character image to check.
            extracted_strokes: Optional pre-extracted strokes (e.g., from perturbation engine)
                               that can be used by graph analyzer.
        Returns:
            True if the image passes all structural integrity checks, False otherwise.
        """
        logger.debug(f"Running structure checks for char '{self.target_char_string}'...")

        # 1. Basic Topology Checks
        if self.basic_checker and self.basic_checker.cfg.enabled:
            if not self.basic_checker.run_checks(image.copy()):
                logger.info("Structure check FAILED: Basic topology check.")
                return False
        else:
            logger.debug("Basic topology checker skipped (disabled or not initialized).")

        # 2. Advanced Topology - Persistent Homology
        if self.ph_analyzer and self.cfg.advanced_topology.enabled and \
                self.cfg.advanced_topology.persistent_homology_params.get("enabled", True):
            if self.reference_ph_signature:
                if not self.ph_analyzer.compare_to_reference_signature(image.copy(), self.reference_ph_signature):
                    logger.info("Structure check FAILED: Persistent homology signature mismatch.")
                    return False
                logger.debug("Persistent homology check passed.")
            else:
                logger.debug("Persistent homology check skipped: No reference signature provided for comparison.")
        else:
            logger.debug("Persistent homology analyzer skipped (disabled or not initialized).")

        # 3. Advanced Topology - Graph Representation
        if self.graph_analyzer and self.cfg.advanced_topology.enabled and \
                self.cfg.advanced_topology.graph_representation_params.get("enabled", True):
            if self.graph_analyzer.reference_graph:  # Check if analyzer has a reference
                if not self.graph_analyzer.check_structure(image.copy(), strokes=extracted_strokes):
                    logger.info("Structure check FAILED: Graph representation mismatch.")
                    return False
                logger.debug("Graph representation check passed.")
            else:
                logger.debug("Graph representation check skipped: No reference graph available for comparison.")
        else:
            logger.debug("Graph representation analyzer skipped (disabled or not initialized).")

        logger.info(f"All enabled structure checks PASSED for char '{self.target_char_string}'.")
        return True


# --- Helper to pre-compute reference signatures (run once or when target char changes) ---
def compute_reference_ph_signature_for_char(char_image_path: str,
                                            adv_topology_cfg: AdvancedTopologyConfig) -> Optional[PersistenceSignature]:
    logger.info(f"Attempting to compute reference PH signature from: {char_image_path}")
    if not os.path.exists(char_image_path):
        logger.error(f"Reference character image not found: {char_image_path}")
        return None

    ref_image = cv2.imread(char_image_path, cv2.IMREAD_UNCHANGED)  # Load as is, PH analyzer handles grayscale
    if ref_image is None:
        logger.error(f"Failed to load reference character image: {char_image_path}")
        return None

    # Ensure PH specific params enable it
    if not adv_topology_cfg.persistent_homology_params.get("enabled", True):
        logger.info("PH reference computation skipped as PH is disabled in config.")
        return None

    ph_analyzer = PersistentHomologyAnalyzer(adv_topology_cfg)
    if not ph_analyzer.gudhi_available and not adv_topology_cfg.persistent_homology_params.get(
            'allow_mock_if_gudhi_unavailable', False):
        logger.warning("Cannot compute reference PH signature: Gudhi unavailable and mock not allowed.")
        return None

    signature = ph_analyzer.get_persistence_signature(ref_image)
    if signature:
        logger.info(f"Computed reference PH signature: {signature}")
    else:
        logger.warning("Failed to compute reference PH signature.")
    return signature


if __name__ == "__main__":
    # --- Test StructureGuard Orchestration ---
    from .....config import SystemConfig  # Adjust for test context

    # Create a reference 'O' image for PH signature
    ref_O_path = "./temp_ref_O_for_structure_guard.png"  # Temp file
    img_O_ref = np.zeros((32, 32), dtype=np.uint8)
    cv2.circle(img_O_ref, (16, 16), 12, (255), thickness=3)
    cv2.imwrite(ref_O_path, img_O_ref)

    temp_sys_cfg_data_sg = {
        "target_character_string": "O",  # Test with 'O'
        "structure_guard": {
            "basic_topology": {  # Config for BasicTopologyChecker
                "enabled": True,
                "rules_for_char": {
                    "O": {"char_threshold": 100, "expected_holes": 1, "min_hole_area": 5}
                }
            },
            "advanced_topology": {  # Config for PHAnalyzer and GraphAnalyzer
                "enabled": True,
                "persistent_homology_params": {
                    "enabled": True, "binarization_threshold": 100,
                    "min_persistence_for_comparison": 30.0, "dim1_feature_tolerance": 0,
                    "allow_mock_if_gudhi_unavailable": True
                },
                "graph_representation_params": {
                    "enabled": False  # Disable graph for this simple test, as it's very conceptual
                }
            }
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_sg)
    cfg_glob_sg = get_config()

    # Compute reference PH signature
    ref_sig = compute_reference_ph_signature_for_char(
        ref_O_path,
        cfg_glob_sg.structure_guard.advanced_topology
    )
    if ref_sig is None and cfg_glob_sg.structure_guard.advanced_topology.persistent_homology_params.get("enabled",
                                                                                                        True):
        print("Failed to compute reference PH signature for test, PH checks might be skipped.")

    # Initialize StructureGuard
    guard = StructureGuard(guard_cfg=cfg_glob_sg.structure_guard,
                           target_char_string=cfg_glob_sg.target_character_string,
                           reference_ph_signature=ref_sig)

    # Test with a good 'O' image
    print("\n--- Testing StructureGuard with a 'good' O image ---")
    test_img_O_good = img_O_ref.copy()
    result_good_O = guard.check_image_structure(test_img_O_good)
    print(f"Structure check for good 'O': {'Pass' if result_good_O else 'Fail'}")
    assert result_good_O

    # Test with a bad 'O' image (e.g., hole filled)
    print("\n--- Testing StructureGuard with a 'bad' O image (hole filled) ---")
    test_img_O_bad = img_O_ref.copy()
    cv2.circle(test_img_O_bad, (16, 16), 8, (255), -1)  # Fill the hole
    # cv2.imshow("Bad O", test_img_O_bad); cv2.waitKey(0)
    result_bad_O = guard.check_image_structure(test_img_O_bad)
    print(f"Structure check for bad 'O': {'Pass' if result_bad_O else 'Fail'}")
    assert not result_bad_O  # Should fail basic hole check or PH check

    # cv2.destroyAllWindows()
    os.remove(ref_O_path)  # Clean up temp ref image
    print("\nStructureGuard tests completed.")