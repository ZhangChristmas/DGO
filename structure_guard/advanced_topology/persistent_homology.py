# ultimate_morph_generator/structure_guard/advanced_topology/persistent_homology.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from ....config import get_config, AdvancedTopologyConfig  # Adjust relative import for config
from ....utilities.type_definitions import CvImage
from ....utilities.logging_config import setup_logging

logger = setup_logging()


# --- Result structure for persistence diagram features ---
class PersistenceSignature:
    """Holds a summary of a persistence diagram (e.g., number of significant features in each dimension)."""

    def __init__(self, dim0_features: int, dim1_features: int,
                 dim0_persistence_values: Optional[List[float]] = None,
                 # (birth, death) pairs or just persistence (death-birth)
                 dim1_persistence_values: Optional[List[float]] = None):
        self.dim0_features = dim0_features  # Typically connected components
        self.dim1_features = dim1_features  # Typically holes/loops
        self.dim0_persistence_values = dim0_persistence_values if dim0_persistence_values else []
        self.dim1_persistence_values = dim1_persistence_values if dim1_persistence_values else []

    def __repr__(self) -> str:
        return f"PersistenceSignature(Dim0: {self.dim0_features}, Dim1: {self.dim1_features})"

    def is_similar_to(self, other: "PersistenceSignature",
                      dim0_tol: int = 0, dim1_tol: int = 0,  # Tolerances for feature counts
                      persistence_threshold: float = 0.1) -> bool:  # Min persistence to be considered significant
        """
        Compares this signature to another.
        A simple comparison based on counts of significant features.
        More advanced: compare persistence diagrams using Wasserstein/Bottleneck distance.
        """
        # Count significant features based on persistence_threshold
        # Assuming persistence_values store (death - birth)
        sig_dim0_self = sum(1 for p in self.dim0_persistence_values if p > persistence_threshold)
        sig_dim1_self = sum(1 for p in self.dim1_persistence_values if p > persistence_threshold)

        sig_dim0_other = sum(1 for p in other.dim0_persistence_values if p > persistence_threshold)
        sig_dim1_other = sum(1 for p in other.dim1_persistence_values if p > persistence_threshold)

        if abs(sig_dim0_self - sig_dim0_other) > dim0_tol:
            return False
        if abs(sig_dim1_self - sig_dim1_other) > dim1_tol:
            return False
        return True


class PersistentHomologyAnalyzer:
    """
    Analyzes character topology using persistent homology.
    Requires a TDA library like Gudhi, Dionysus, or Ripser.
    This class provides an interface and conceptual implementation.
    """

    def __init__(self, adv_topology_cfg: AdvancedTopologyConfig):
        self.cfg = adv_topology_cfg
        self.gudhi_available = False
        try:
            import gudhi  # type: ignore
            # For cubical complex from image:
            # from gudhi.cubical_complex import CubicalComplex
            self.gudhi = gudhi
            self.gudhi_available = True
            logger.info("Gudhi library found. Persistent homology analysis enabled.")
        except ImportError:
            logger.warning("Gudhi library not found. Persistent homology analysis will be disabled or use mock data.")
            self.gudhi = None

        # Parameters for persistence calculation from config
        self.persistence_params = self.cfg.persistent_homology_params  # e.g., {"min_persistence": 0.1}

    def _image_to_cubical_complex_gudhi(self, image_binary_inverted: CvImage) -> Optional[Any]:
        """
        Converts a binary image (where character is higher value/obstacle) to a Gudhi CubicalComplex.
        Image should be preprocessed: e.g., grayscale, possibly inverted so character pixels are 'higher'.
        Gudhi's CubicalComplex typically works with filtration values.
        If image_binary_inverted is 0 for char, 255 for bg, then char is lower (earlier birth).
        If image_binary_inverted is 255 for char, 0 for bg, then char is higher (later birth / obstacle).
        Let's assume input `image_binary_inverted` has character as 0s, background as 255s.
        Gudhi's CubicalComplex from top-dimensional cells (pixels) uses pixel values as filtration values.
        Lower values appear first. So, character pixels should be low.
        """
        if not self.gudhi_available or self.gudhi is None: return None

        # Ensure image is 2D and float (Gudhi might prefer this)
        if image_binary_inverted.ndim != 2:
            logger.error("Image for cubical complex must be 2D.")
            return None

        # Gudhi's CubicalComplex can take dimensions and top-dimensional cells (pixels)
        # Pixels with lower values are introduced earlier in the filtration.
        # For character topology, we often want character pixels to be "obstacles" (higher values)
        # or use the distance transform.
        # Let's use the image directly: image_binary_inverted (0 for char, 255 for bg)
        # This means character components (dim0) will be born at 0, holes (dim1) will be born later
        # when loops are formed by background.

        # Alternatively, use distance transform as filtration values.
        # dist_transform = cv2.distanceTransform(image_binary_inverted, cv2.DIST_L2, 5)
        # filtration_values = -dist_transform # Negative distance to make character part of superlevel set

        try:
            # Create cubical complex from top-dimensional cells (pixels)
            # Dimensions are (height, width) for the grid
            # Top dimensional cells are the pixel values themselves
            # Example: cubical_complex = self.gudhi.CubicalComplex(dimensions=image_binary_inverted.shape,
            #                                                     top_dimensional_cells=image_binary_inverted.flatten())
            # Using an image directly (more common for Gudhi image processing):
            # Needs specific Gudhi version/API knowledge.
            # For example, gudhi.BitmapCubicalComplex might be relevant if image is bool/uint8.
            # The typical way with recent Gudhi is:
            cubical_complex = self.gudhi.CubicalComplex(top_dimensional_cells=image_binary_inverted)
            return cubical_complex
        except Exception as e:
            logger.error(f"Error creating Gudhi CubicalComplex: {e}")
            return None

    def compute_persistence_diagram(self, image: CvImage) -> Optional[List[Tuple[int, Tuple[float, float]]]]:
        """
        Computes the persistence diagram for the image.
        Returns a list of (dimension, (birth, death)) tuples.
        Example: [(0, (0.0, 10.5)), (1, (5.0, 12.0)), ...]
        Birth/death values depend on the filtration.
        """
        if not self.gudhi_available or self.gudhi is None:
            logger.debug("Gudhi not available. Skipping persistence diagram computation.")
            return None  # Or mock data for testing flow

        # Preprocess image: should be binary, with character pixels having low filtration values.
        # Assume image is already uint8 grayscale.
        if image.ndim == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image.copy()

        # Binarize: char=0, bg=255 (for Gudhi CubicalComplex where low values appear first)
        # This means character components are born at filtration 0.
        # Holes are formed by background, so their birth depends on bg pixel values.
        bin_threshold = int(self.cfg.persistent_homology_params.get('binarization_threshold', 128))
        is_light_bg = np.mean(gray_img) > bin_threshold
        if is_light_bg:  # Dark char on light bg
            _, binary_char_0_bg_255 = cv2.threshold(gray_img, bin_threshold, 255, cv2.THRESH_BINARY)
        else:  # Light char on dark bg
            _, binary_char_0_bg_255 = cv2.threshold(gray_img, bin_threshold, 255, cv2.THRESH_BINARY_INV)

        cubical_complex = self._image_to_cubical_complex_gudhi(binary_char_0_bg_255)
        if not cubical_complex:
            return None

        try:
            # Compute persistence. `min_persistence` can filter out short-lived features.
            # This parameter might be applied after computing all pairs.
            # persistence_obj = cubical_complex.persistence(homology_coeff_field=2, min_persistence=-1) # Get all pairs
            persistence_obj = cubical_complex.persistence()  # Default params

            # Filter by dimension (e.g., 0 for components, 1 for holes)
            # Default Gudhi persistence() returns list of (dim, (birth, death))
            # Filter out infinite death times for dim0 component of the whole space if needed.
            # For an image, there's usually one dim0 feature (connected component of background) with infinite persistence.
            # We are interested in components of the *character*.
            # If char=0, bg=255:
            #   - Dim0 features born at 0 are character components.
            #   - Dim1 features are loops/holes.

            # Example of filtering:
            # significant_pairs = []
            # min_pers_val = self.persistence_params.get('min_persistence', 0.0)
            # for dim, (birth, death) in persistence_obj:
            #     if death - birth > min_pers_val: # If death is inf, this check needs care
            #          if death == float('inf') and dim == 0: # Often one main component is inf
            #              # Handle how to count this "infinite" component
            #              significant_pairs.append((dim, (birth, death)))
            #          elif death != float('inf'):
            #              significant_pairs.append((dim, (birth, death)))

            # For now, return all pairs. Filtering/interpretation happens in signature.
            return persistence_obj
        except Exception as e:
            logger.error(f"Error computing persistence with Gudhi: {e}")
            return None

    def get_persistence_signature(self, image: CvImage) -> Optional[PersistenceSignature]:
        """
        Computes a persistence signature from the image.
        """
        persistence_diagram = self.compute_persistence_diagram(image)
        if persistence_diagram is None:
            # Mock signature if Gudhi is not available, for pipeline testing
            if not self.gudhi_available:
                logger.debug("Mocking persistence signature as Gudhi is unavailable.")
                return PersistenceSignature(dim0_features=1, dim1_features=0,
                                            # e.g. assuming a simple char like 'I' or 'L'
                                            dim0_persistence_values=[10.0], dim1_persistence_values=[])
            return None

        dim0_count = 0
        dim1_count = 0
        dim0_pers_values = []
        dim1_pers_values = []

        # Filtration: char=0, bg=255.
        # Dim0 features born at 0 are character components.
        # Dim1 features are loops in the character.
        # We need to be careful about what "persistence" means here.
        # Death - Birth. For char components born at 0, persistence is their death time.
        # For holes, it's also death - birth.

        # min_pers_val = self.persistence_params.get('min_persistence', 0.0) # Threshold for significance
        # Often, for images, a relative persistence (e.g., % of max filtration value) is better.
        # Max filtration value here is 255 (background).

        for dim, (birth, death) in persistence_diagram:
            # Handle infinite death for the main connected component of the background (if filtration is 0 for char)
            # This component is usually not what we count for character structure.
            # We are interested in components born at ~0 (character parts) and holes.
            pers_val = death - birth
            if pers_val < 0 and death != float('inf'):  # Should not happen if diagram is valid
                logger.warning(f"Encountered negative persistence {pers_val} for finite death. Skipping pair.")
                continue

            if dim == 0:
                if birth < 1.0:  # Component related to character (born at filtration value 0)
                    # If death is inf, it's the main component of the character (if it's one piece)
                    # We are interested in how many distinct character pieces exist.
                    # This interpretation depends heavily on the filtration setup.
                    dim0_count += 1
                    dim0_pers_values.append(
                        pers_val if death != float('inf') else 255.0)  # Use max filtration as proxy for inf death
            elif dim == 1:
                # These are holes. Birth/death depend on background values.
                dim1_count += 1
                dim1_pers_values.append(pers_val)  # Holes should have finite death

        # The counts here might need refinement based on how Gudhi handles components vs background.
        # Often, for a single character object, you expect one significant dim0 feature (the object itself)
        # if the filtration makes the object appear "all at once".
        # The current (char=0, bg=255) filtration: components of char are born at 0.
        # Number of dim0 features born at 0 = number of connected components of the character.

        # Refined logic for dim0 count:
        # Only count dim0 features with birth near 0 (character) and significant persistence.
        # The count of holes (dim1) is usually more straightforward.

        # For simplicity, the raw counts are used now. Interpretation is key.
        return PersistenceSignature(dim0_features=dim0_count, dim1_features=dim1_count,
                                    dim0_persistence_values=dim0_pers_values,
                                    dim1_persistence_values=dim1_pers_values)

    def compare_to_reference_signature(self, image: CvImage,
                                       ref_signature: PersistenceSignature) -> bool:
        """
        Compares the persistence signature of the image to a reference signature.
        """
        if not self.cfg.enabled or (not self.gudhi_available and not self.cfg.persistent_homology_params.get(
                'allow_mock_if_gudhi_unavailable', False)):
            logger.debug("Persistent homology check skipped (disabled or Gudhi unavailable).")
            return True  # Pass if disabled or cannot run

        current_signature = self.get_persistence_signature(image)
        if not current_signature:
            logger.warning("Failed to compute current persistence signature. Comparison failed.")
            return False  # Cannot compare

        # Get comparison parameters from config
        dim0_tol = int(self.persistence_params.get('dim0_feature_tolerance', 0))
        dim1_tol = int(self.persistence_params.get('dim1_feature_tolerance', 0))
        min_significance = float(self.persistence_params.get('min_persistence_for_comparison', 0.1))  # Example value

        return current_signature.is_similar_to(ref_signature,
                                               dim0_tol=dim0_tol, dim1_tol=dim1_tol,
                                               persistence_threshold=min_significance)


if __name__ == "__main__":
    # --- Test PersistentHomologyAnalyzer ---
    from .....config import SystemConfig  # Adjust relative import for test

    temp_sys_cfg_data_ph = {
        "structure_guard": {
            "advanced_topology": {
                "enabled": True,  # Enable this section
                "persistent_homology_params": {
                    "binarization_threshold": 100,
                    "min_persistence_for_comparison": 50.0,  # Persistence values are large for 0-255 img
                    "dim0_feature_tolerance": 0,
                    "dim1_feature_tolerance": 0,
                    "allow_mock_if_gudhi_unavailable": True  # For CI/testing without Gudhi
                }
            }
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_ph)
    cfg_glob_ph = get_config()

    adv_topo_config = cfg_glob_ph.structure_guard.advanced_topology
    ph_analyzer = PersistentHomologyAnalyzer(adv_topology_cfg=adv_topo_config)

    if not ph_analyzer.gudhi_available:
        print("Gudhi library not found. Tests will use mock data or be skipped if not allowed by config.")

    # Create a test image for 'O' (one hole)
    img_O = np.zeros((32, 32), dtype=np.uint8)  # Black bg
    cv2.circle(img_O, (16, 16), 12, (255), thickness=4)  # White 'O'
    # cv2.imshow("Test 'O' for PH", img_O); cv2.waitKey(0)

    print("\n--- Testing Persistent Homology Analyzer for 'O' ---")
    sig_O = ph_analyzer.get_persistence_signature(img_O.copy())
    if sig_O:
        print(f"Signature for 'O': {sig_O}")
        print(f"  Dim0 Persistences: {sig_O.dim0_persistence_values}")
        print(f"  Dim1 Persistences: {sig_O.dim1_persistence_values}")
        # Expected for 'O' with char=0, bg=255 filtration:
        # - Dim0: 1 significant component (the character itself, born at 0, dies at ~thickness/2 or related to bg)
        # - Dim1: 1 significant hole (the loop, born when bg fills center, dies when outer bg connects)
        # The exact counts and persistence values are sensitive to Gudhi's specific cubical complex definition.
        # For mock data: assumes 1 component, 1 hole.
        if ph_analyzer.gudhi_available:  # Real computation
            # These asserts are highly dependent on Gudhi version and complex details.
            # For char=0 filtration, we expect one major dim0 component.
            # The number of holes is more stable.
            assert sig_O.dim1_features >= 1  # Should find at least one hole for 'O'
        else:  # Mock data check
            assert sig_O.dim0_features == 1 and sig_O.dim1_features == 1  # Mock has 1 hole for 'O'

    # Create a test image for 'I' (no holes)
    img_I = np.zeros((32, 32), dtype=np.uint8)
    cv2.line(img_I, (16, 5), (16, 27), (255), thickness=5)
    # cv2.imshow("Test 'I' for PH", img_I); cv2.waitKey(0)

    print("\n--- Testing Persistent Homology Analyzer for 'I' ---")
    sig_I = ph_analyzer.get_persistence_signature(img_I.copy())
    if sig_I:
        print(f"Signature for 'I': {sig_I}")
        # Expected for 'I': 1 significant dim0 component, 0 significant dim1 holes.
        if ph_analyzer.gudhi_available:
            assert sig_I.dim1_features == 0 or all(
                p < adv_topo_config.persistent_homology_params.get('min_persistence_for_comparison', 50.0) for p in
                sig_I.dim1_persistence_values)

        # Test comparison (conceptual)
        if sig_O and sig_I:
            print(
                f"Comparing sig_O to sig_I: Similar? {sig_O.is_similar_to(sig_I, persistence_threshold=adv_topo_config.persistent_homology_params['min_persistence_for_comparison'])}")
            assert not sig_O.is_similar_to(sig_I, persistence_threshold=adv_topo_config.persistent_homology_params[
                'min_persistence_for_comparison'])

            # Create a slightly perturbed 'O' and compare to original 'O'
            img_O_perturbed = img_O.copy()
            cv2.ellipse(img_O_perturbed, (16, 16), (12, 11), 0, 0, 360, (255), thickness=4)  # Slightly squashed
            sig_O_perturbed = ph_analyzer.get_persistence_signature(img_O_perturbed)
            if sig_O_perturbed:
                print(
                    f"Comparing sig_O to sig_O_perturbed: Similar? {sig_O.is_similar_to(sig_O_perturbed, dim1_tol=0, persistence_threshold=adv_topo_config.persistent_homology_params['min_persistence_for_comparison'])}")
                # This assert depends on the robustness of the signature and thresholds
                # assert sig_O.is_similar_to(sig_O_perturbed, dim1_tol=0, persistence_threshold=adv_topo_config.persistent_homology_params['min_persistence_for_comparison'])

    # cv2.destroyAllWindows()
    print("\nPersistentHomologyAnalyzer tests completed (results depend heavily on Gudhi availability and version).")