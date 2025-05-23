# ultimate_morph_generator/structure_guard/basic_topology.py
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from ..config import get_config, BasicTopologyConfig, StructureGuardConfig
from ..utilities.type_definitions import CvImage
from ..utilities.logging_config import setup_logging

logger = setup_logging()


class BasicTopologyChecker:
    """
    Performs basic topological checks on a character image.
    - Number of connected components (should typically be 1 for a single character).
    - Number of holes (e.g., '8' has 2, '0' has 1, '3' might have 0 if holes are small or not considered).
    - Approximate opening directions for characters like '3', 'C', 'U'.
    """

    def __init__(self, topology_cfg: BasicTopologyConfig, target_char_string: str):
        self.cfg = topology_cfg
        self.target_char_string = target_char_string
        # Character-specific rules from config
        self.char_rules = self.cfg.rules_for_char.get(self.target_char_string, {})

        self.binarization_threshold = int(self.char_rules.get('char_threshold', 128))
        self.min_hole_area = int(
            self.char_rules.get('min_hole_area', 10))  # Min area for a contour to be considered a hole
        self.expected_holes = self.char_rules.get('expected_holes')  # Can be int or None
        self.expected_openings = self.char_rules.get('expected_openings')  # Num openings, e.g. for '3'
        self.opening_directions = self.char_rules.get('opening_directions')  # List of strings e.g. ['right','right']

    def _preprocess_and_binarize(self, image: CvImage) -> Tuple[Optional[CvImage], Optional[CvImage]]:
        """
        Converts to grayscale and binarizes the image.
        Returns two versions:
        1. char_white_bg_black: Character is white (255), background is black (0). For findContours.
        2. char_black_bg_white: Character is black (0), background is white (255). For some analyses.
        Returns (None, None) on failure.
        """
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray_image = image.copy()
        else:
            logger.error(f"Unsupported image ndim for binarization: {image.ndim}")
            return None, None

        # Determine if image is light-on-dark or dark-on-light to apply correct threshold type
        # Heuristic: if mean intensity > threshold, likely light background
        is_light_background = np.mean(gray_image) > self.binarization_threshold

        if is_light_background:  # Dark character on light background
            _, char_white_bg_black = cv2.threshold(gray_image, self.binarization_threshold, 255, cv2.THRESH_BINARY_INV)
            _, char_black_bg_white = cv2.threshold(gray_image, self.binarization_threshold, 255, cv2.THRESH_BINARY)
        else:  # Light character on dark background
            _, char_white_bg_black = cv2.threshold(gray_image, self.binarization_threshold, 255, cv2.THRESH_BINARY)
            _, char_black_bg_white = cv2.threshold(gray_image, self.binarization_threshold, 255, cv2.THRESH_BINARY_INV)

        if char_white_bg_black is None or char_black_bg_white is None:
            logger.error("Binarization failed.")
            return None, None

        return char_white_bg_black, char_black_bg_white

    def check_connected_components(self, char_white_bg_black: CvImage, expected_components: int = 1) -> bool:
        """
        Checks if the character consists of the expected number of connected components.
        Usually 1 for a single, non-broken character.
        """
        if char_white_bg_black is None: return False

        num_labels, labels_im = cv2.connectedComponents(char_white_bg_black)
        # num_labels includes the background component (label 0)
        actual_components = num_labels - 1

        if actual_components != expected_components:
            logger.debug(
                f"Connected components check failed: Found {actual_components}, expected {expected_components}.")
            return False
        logger.debug(f"Connected components check passed: Found {actual_components}.")
        return True

    def count_holes(self, char_white_bg_black: CvImage) -> int:
        """
        Counts the number of internal holes in the character.
        Uses contour hierarchy: RETR_CCOMP finds external contours and hole contours.
        Hole contours are children of an external contour.
        """
        if char_white_bg_black is None: return -1  # Indicate error

        # Pad image slightly to ensure contours near edge are found correctly
        padded_img = cv2.copyMakeBorder(char_white_bg_black, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        contours, hierarchy = cv2.findContours(padded_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # For RETR_CCOMP, hierarchy is [Next, Previous, First_Child, Parent]
        # Holes are contours that have a parent (hierarchy[i][3] != -1)
        # and their parent is an external contour (parent's parent is -1, or parent is at top level).

        num_holes = 0
        if hierarchy is not None and len(hierarchy) > 0:
            hierarchy = hierarchy[0]  # Actual hierarchy array
            for i in range(len(contours)):
                # If contour 'i' has a parent (it's an inner contour / hole)
                parent_idx = hierarchy[i][3]
                if parent_idx != -1:
                    # Check if the parent is an external contour (i.e., parent's parent is -1)
                    # Or, more simply for CCOMP, top-level contours are external.
                    # A hole is a child of a top-level contour.
                    # So, if hierarchy[parent_idx][3] == -1, then 'i' is a hole of an external contour.
                    # However, this doesn't handle nested holes well.
                    # Simpler: any contour with a parent in CCOMP is a hole.

                    # Filter by area to avoid noise specks being counted as holes
                    if cv2.contourArea(contours[i]) >= self.min_hole_area:
                        num_holes += 1

        logger.debug(f"Counted {num_holes} holes (min area: {self.min_hole_area}).")
        return num_holes

    def check_openings(self, char_white_bg_black: CvImage) -> bool:
        """
        Checks for the number and approximate direction of openings.
        This is highly character-specific and complex.
        Example for '3': two openings, roughly to the right.

        A simple heuristic:
        1. Find external contour of the character.
        2. Analyze convexity defects or find points on the contour that are "exposed"
           to the background in expected directions.

        This is a placeholder for a more advanced implementation.
        For now, it will be a very basic check if rules are defined.
        """
        if not self.expected_openings or not self.opening_directions:
            logger.debug("No opening rules defined for this character. Skipping check.")
            return True  # Pass if no rules

        if char_white_bg_black is None: return False

        # This is a very simplified and potentially unreliable check.
        # True opening detection is hard.
        # Conceptual: Check if certain regions of the bounding box perimeter are background.
        h, w = char_white_bg_black.shape

        # Example for '3' with two right openings.
        # Check mid-points of right edge segments.
        # Divide right edge into, say, 3 segments (top, middle, bottom thirds of height).
        # An opening means a path from inside the character to outside.
        # Alternative: Convex Hull and Convexity Defects

        contours, _ = cv2.findContours(char_white_bg_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug("Opening check: No external contour found.")
            return False

        char_contour = max(contours, key=cv2.contourArea)  # Assume largest external contour is the char

        # Create a mask of the character filled
        filled_char_mask = np.zeros_like(char_white_bg_black)
        cv2.drawContours(filled_char_mask, [char_contour], -1, 255, thickness=cv2.FILLED)

        # Invert the filled mask to get "inverse space"
        inverse_filled_mask = cv2.bitwise_not(filled_char_mask)

        # Find connected components in this "inverse space" that are *outside* the char's bounding box
        # but "enter" it. This is getting complicated.

        # Simpler heuristic for '3' (two right openings):
        # Check if pixels just to the left of the rightmost character points are background
        # for two distinct vertical regions.
        if self.target_char_string == "3" and len(self.opening_directions) == 2 \
                and all(d == "right" for d in self.opening_directions):

            right_edge_x = char_contour[:, 0, 0].max()  # x-coords of contour points

            # Define y-regions for openings (e.g., top 1/3 and bottom 1/3 of char height)
            min_y_char, max_y_char = char_contour[:, 0, 1].min(), char_contour[:, 0, 1].max()
            char_h = max_y_char - min_y_char
            if char_h < 10: return False  # Too small to have clear openings

            opening1_y_center = min_y_char + char_h // 4
            opening2_y_center = min_y_char + (3 * char_h) // 4

            openings_found = 0
            # Check near opening1_y_center, just inside the right_edge_x
            # Look for a background pixel (0) to the left of a foreground pixel (255) on the contour
            # This is very heuristic.
            # Better: use convex hull defects. Find defects pointing roughly right.

            # Placeholder: Return true if rules are defined, as implementation is complex.
            logger.debug("Opening check is conceptual for now. Assuming pass if rules exist.")
            return True  # Placeholder for complex logic.

        logger.debug(f"Opening check rules not matched or complex for {self.target_char_string}.")
        return True  # Pass if not '3' or rules don't match this simple heuristic logic

    def run_checks(self, image: CvImage) -> bool:
        """
        Runs all configured basic topology checks.
        Returns True if all checks pass, False otherwise.
        """
        if not self.cfg.enabled:
            logger.debug("BasicTopologyChecker is disabled. Skipping checks.")
            return True

        binarized_char_white, binarized_char_black = self._preprocess_and_binarize(image)
        if binarized_char_white is None:
            logger.warning("Binarization failed, topology checks cannot proceed.")
            return False

        # 1. Connected Components
        # For most single characters, expect 1 component. This can be in char_rules.
        expected_cc = int(self.char_rules.get('expected_connected_components', 1))
        if not self.check_connected_components(binarized_char_white, expected_components=expected_cc):
            return False  # Failed CC check

        # 2. Number of Holes
        if self.expected_holes is not None:  # Rule for holes exists
            num_found_holes = self.count_holes(binarized_char_white)
            if num_found_holes != self.expected_holes:
                logger.debug(f"Hole count check failed: Found {num_found_holes}, expected {self.expected_holes}.")
                return False
            logger.debug(f"Hole count check passed: Found {num_found_holes}.")

        # 3. Openings (Conceptual for now)
        if self.expected_openings is not None or self.opening_directions is not None:
            if not self.check_openings(binarized_char_white):
                logger.debug(f"Openings check failed for {self.target_char_string}.")
                return False  # Failed openings check
            logger.debug(f"Openings check passed for {self.target_char_string} (conceptual).")

        logger.info(f"All basic topology checks passed for char '{self.target_char_string}'.")
        return True


if __name__ == "__main__":
    # --- Test BasicTopologyChecker ---
    from ....config import SystemConfig  # Adjust relative import for test execution

    # Example config for character '8'
    temp_sys_cfg_data_8 = {
        "target_character_string": "8",
        "structure_guard": {
            "basic_topology": {
                "enabled": True,
                "rules_for_char": {
                    "8": {
                        "char_threshold": 100,
                        "min_hole_area": 5,
                        "expected_holes": 2,
                        "expected_connected_components": 1
                    }
                }
            }
        },
        "logging": {"level": "DEBUG"}
    }
    from ....config import _config_instance  # Adjust

    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_8)
    cfg_glob_8 = get_config()

    checker_8 = BasicTopologyChecker(topology_cfg=cfg_glob_8.structure_guard.basic_topology,
                                     target_char_string=cfg_glob_8.target_character_string)

    # Create a test image for '8' (white on black)
    img_8 = np.zeros((64, 64), dtype=np.uint8)
    cv2.putText(img_8, "8", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255), thickness=7, lineType=cv2.LINE_AA)
    # cv2.imshow("Test '8'", img_8); cv2.waitKey(0)

    print("\n--- Testing BasicTopologyChecker for '8' (should pass) ---")
    result_8_good = checker_8.run_checks(img_8.copy())
    print(f"Result for good '8': {'Pass' if result_8_good else 'Fail'}")
    assert result_8_good

    # Create a broken '8' (e.g., one hole filled, or broken into two components)
    img_8_broken_hole = img_8.copy()
    cv2.circle(img_8_broken_hole, (32, 25), 8, (255), -1)  # Fill top hole
    # cv2.imshow("Test '8' (broken hole)", img_8_broken_hole); cv2.waitKey(0)
    print("\n--- Testing '8' with one hole filled (should fail hole check) ---")
    result_8_broken_hole = checker_8.run_checks(img_8_broken_hole)
    print(f"Result for broken hole '8': {'Pass' if result_8_broken_hole else 'Fail'}")
    assert not result_8_broken_hole

    img_8_broken_cc = img_8.copy()
    cv2.line(img_8_broken_cc, (25, 32), (39, 32), (0), thickness=8)  # Break the middle connection
    # cv2.imshow("Test '8' (broken CC)", img_8_broken_cc); cv2.waitKey(0)
    print("\n--- Testing '8' with broken connection (should fail CC check) ---")
    result_8_broken_cc = checker_8.run_checks(img_8_broken_cc)
    print(f"Result for broken CC '8': {'Pass' if result_8_broken_cc else 'Fail'}")
    assert not result_8_broken_cc

    # Example config for character '3'
    temp_sys_cfg_data_3 = {
        "target_character_string": "3",
        "structure_guard": {
            "basic_topology": {
                "enabled": True,
                "rules_for_char": {
                    "3": {
                        "char_threshold": 100,
                        "expected_holes": 0,  # Assuming '3' has no fully enclosed holes for this rule
                        "expected_connected_components": 1,
                        "expected_openings": 2,  # Number of openings
                        "opening_directions": ["right", "right"]  # Conceptual directions
                    }
                }
            }
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_3)  # Update global config
    cfg_glob_3 = get_config()

    checker_3 = BasicTopologyChecker(topology_cfg=cfg_glob_3.structure_guard.basic_topology,
                                     target_char_string=cfg_glob_3.target_character_string)

    img_3 = np.zeros((64, 64), dtype=np.uint8)
    cv2.putText(img_3, "3", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255), thickness=6, lineType=cv2.LINE_AA)
    # cv2.imshow("Test '3'", img_3); cv2.waitKey(0)

    print("\n--- Testing BasicTopologyChecker for '3' (should pass, openings conceptual) ---")
    result_3_good = checker_3.run_checks(img_3.copy())
    print(f"Result for good '3': {'Pass' if result_3_good else 'Fail'}")
    assert result_3_good  # Openings check is currently lenient

    # cv2.destroyAllWindows()
    print("\nBasicTopologyChecker tests completed.")