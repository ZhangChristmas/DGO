# ultimate_morph_generator/perturbation_suite/stroke_engine/stroke_perturbations.py
import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional

from .stroke_extractor import Stroke  # Uses the Stroke class
from ...config import get_config, StrokeEngineConfig  # For parameters
from ...utilities.type_definitions import CvImage
from ...utilities.logging_config import setup_logging

logger = setup_logging()


class StrokePerturber:
    """
    Applies perturbations to character images based on their extracted strokes.
    """

    def __init__(self, stroke_cfg: StrokeEngineConfig, target_image_size: Tuple[int, int]):
        self.cfg = stroke_cfg
        self.target_image_size = target_image_size  # H, W
        # Define ranges for perturbation parameters from stroke_cfg if available
        # Example: self.thickness_change_range = getattr(self.cfg, 'thickness_change_range', (-2, 2))
        #          self.point_displacement_std = getattr(self.cfg, 'point_displacement_std', 1.0)
        # For now, use hardcoded defaults or simple ranges.
        self.thickness_change_abs_max = getattr(self.cfg, 'thickness_change_abs_max', 3)  # Max pixels to add/remove
        self.point_displacement_scale = getattr(self.cfg, 'point_displacement_scale', 0.05)  # Proportion of image dim
        self.control_point_bend_scale = getattr(self.cfg, 'control_point_bend_scale', 0.1)
        self.stroke_selection_prob = getattr(self.cfg, 'perturbation_probability',
                                             0.5)  # Prob of perturbing any given stroke

    def _render_strokes_to_image(self, strokes: List[Stroke], image_size: Tuple[int, int],
                                 background_color: int = 0, default_fg_color: int = 255) -> CvImage:
        """
        Renders a list of Stroke objects back into an image.
        Assumes strokes have points and avg_thickness.
        """
        canvas = np.full(image_size, background_color, dtype=np.uint8)

        for stroke in strokes:
            if len(stroke.points) < 2:
                continue

            thickness = stroke.avg_thickness
            if thickness is None or thickness <= 0:
                thickness = max(1, int(min(image_size) * 0.05))  # Default thickness if not available
            thickness = int(round(max(1, thickness)))  # Ensure at least 1px

            for i in range(len(stroke.points) - 1):
                p1 = tuple(stroke.points[i].astype(int))
                p2 = tuple(stroke.points[i + 1].astype(int))
                try:
                    cv2.line(canvas, p1, p2, (default_fg_color), thickness=thickness, lineType=cv2.LINE_AA)
                except cv2.error as e:  # Points might be out of bounds after perturbation
                    # logger.warning(f"cv2.line error rendering stroke {stroke.id}: {e}. Points: {p1}, {p2}")
                    # Clip points to be within image bounds for drawing
                    h, w = image_size
                    p1_clipped = (np.clip(p1[0], 0, w - 1), np.clip(p1[1], 0, h - 1))
                    p2_clipped = (np.clip(p2[0], 0, w - 1), np.clip(p2[1], 0, h - 1))
                    if p1_clipped != p2_clipped:  # Avoid drawing line if points collapse
                        try:
                            cv2.line(canvas, p1_clipped, p2_clipped, (default_fg_color), thickness=thickness,
                                     lineType=cv2.LINE_AA)
                        except Exception as inner_e:
                            logger.error(f"Error drawing clipped line for stroke {stroke.id}: {inner_e}")
        return canvas

    def perturb_stroke_thickness(self, stroke: Stroke) -> Stroke:
        """Perturbs the thickness of a single stroke."""
        if stroke.avg_thickness is not None:
            # Change can be relative or absolute. For simplicity, absolute change here.
            change = random.randint(-self.thickness_change_abs_max, self.thickness_change_abs_max)
            stroke.avg_thickness = max(1.0, stroke.avg_thickness + change)  # Ensure thickness >= 1
        return stroke

    def perturb_stroke_points_local(self, stroke: Stroke) -> Stroke:
        """Slightly displaces points along the stroke locally (like jitter)."""
        # Max displacement related to image size or a fixed pixel value
        max_disp = max(1, int(min(self.target_image_size) * self.point_displacement_scale))

        displacements_x = np.random.randint(-max_disp, max_disp + 1, size=stroke.points.shape[0])
        displacements_y = np.random.randint(-max_disp, max_disp + 1, size=stroke.points.shape[0])

        perturbed_points = stroke.points.copy()
        perturbed_points[:, 0] += displacements_x
        perturbed_points[:, 1] += displacements_y

        # Clip points to image boundaries (optional, can be done at rendering)
        # h, w = self.target_image_size
        # perturbed_points[:, 0] = np.clip(perturbed_points[:, 0], 0, w - 1)
        # perturbed_points[:, 1] = np.clip(perturbed_points[:, 1], 0, h - 1)
        stroke.points = perturbed_points
        return stroke

    def perturb_stroke_shape_bend(self, stroke: Stroke) -> Stroke:
        """
        Bends a stroke by displacing a few control points (e.g., midpoint).
        This is a simplified bending. True spline/Bezier manipulation is more complex.
        """
        if len(stroke.points) < 3:  # Need at least 3 points to define a bend meaningfully
            return stroke

        # Select a few "control points" to move (e.g., midpoint, quarter points)
        # For simplicity, let's try to displace the middle point of the stroke.
        mid_idx = len(stroke.points) // 2

        # Calculate a displacement vector, perhaps perpendicular to the stroke segment at midpoint.
        # Simplified: random displacement.
        max_bend_disp = max(1, int(min(self.target_image_size) * self.control_point_bend_scale))
        disp_x = random.randint(-max_bend_disp, max_bend_disp + 1)
        disp_y = random.randint(-max_bend_disp, max_bend_disp + 1)

        perturbed_points = stroke.points.copy()
        perturbed_points[mid_idx, 0] += disp_x
        perturbed_points[mid_idx, 1] += disp_y

        # TODO: For smoother bending, one would interpolate new points based on moved control points
        # (e.g., using quadratic/cubic interpolation if only a few points define the original stroke,
        # or by re-fitting a spline if the stroke is dense).
        # This simplified version just moves one point.
        stroke.points = perturbed_points
        return stroke

    # More advanced perturbations:
    # - perturb_stroke_length (extend/shorten at endpoints)
    # - perturb_stroke_connectivity (break/join strokes at junctions - very hard)
    # - perturb_stroke_angle (rotate a stroke around a pivot, e.g., its start or center)

    def apply_stroke_perturbations_to_list(self, strokes: List[Stroke]) -> List[Stroke]:
        """
        Applies a random selection of perturbations to a list of strokes.
        Each stroke has a `self.stroke_selection_prob` chance of being perturbed.
        If perturbed, one of the available perturbation types is chosen randomly.
        """
        perturbed_strokes: List[Stroke] = []
        available_perturb_fns = [
            self.perturb_stroke_thickness,
            self.perturb_stroke_points_local,
            self.perturb_stroke_shape_bend
        ]

        for original_stroke in strokes:
            # Create a copy to modify, so original list isn't changed if passed by reference elsewhere
            current_stroke = Stroke.from_dict(original_stroke.to_dict())  # Deep copy via dict

            if random.random() < self.stroke_selection_prob:
                # Choose a perturbation function to apply to this stroke
                perturb_fn = random.choice(available_perturb_fns)
                logger.debug(f"Applying stroke perturbation '{perturb_fn.__name__}' to stroke {current_stroke.id}")
                current_stroke = perturb_fn(current_stroke)

            perturbed_strokes.append(current_stroke)

        return perturbed_strokes

    def perturb_image_via_strokes(self, image: CvImage, extracted_strokes: List[Stroke]) -> CvImage:
        """
        High-level function: Takes an image and its extracted strokes,
        perturbs the strokes, and renders them back to a new image.
        """
        if not extracted_strokes:
            logger.warning("No strokes provided for perturbation. Returning original image.")
            return image.copy()

        # 1. Apply perturbations to the Stroke objects
        perturbed_stroke_objects = self.apply_stroke_perturbations_to_list(extracted_strokes)

        # 2. Render the perturbed strokes back to an image
        # Determine background color from original image (e.g. corners)
        # For simplicity, assume black background, white foreground from config or typical char images
        bg_color = 0
        fg_color = 255
        # Heuristic to guess if input is inverted (dark on light)
        # if np.mean(image) > 128: bg_color, fg_color = 255, 0

        final_perturbed_image = self._render_strokes_to_image(
            perturbed_stroke_objects,
            image_size=image.shape[:2],  # Use original image size (H,W)
            background_color=bg_color,
            default_fg_color=fg_color
        )

        return final_perturbed_image


if __name__ == "__main__":
    # --- Test StrokePerturber ---
    # Needs StrokeExtractor to provide strokes first.
    from .stroke_extractor import StrokeExtractor  # For test dependency
    from ...config import SystemConfig  # Adjust relative import

    # Config for stroke engine
    temp_sys_cfg_data_stroke_eng = {
        "perturbation_suite": {
            "stroke_engine_perturbations": {  # This is StrokeEngineConfig
                "enabled": True,
                "extractor_type": "skeletonization_vectorization",
                "min_stroke_length": 3,
                "perturbation_probability": 0.8,  # Prob of perturbing a stroke
                "thickness_change_abs_max": 2,  # Custom param for StrokePerturber
                "point_displacement_scale": 0.03,
                "control_point_bend_scale": 0.08
            }
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_stroke_eng)
    cfg_glob = get_config()
    stroke_engine_config = cfg_glob.perturbation_suite.stroke_engine_perturbations

    # Create extractor and perturber
    extractor = StrokeExtractor(stroke_cfg=stroke_engine_config)
    extractor.binarization_threshold = 100

    img_size_test = (48, 48)  # H,W
    perturber = StrokePerturber(stroke_cfg=stroke_engine_config, target_image_size=img_size_test)

    # Create a simple test image (letter 'L')
    test_img_L = np.zeros(img_size_test, dtype=np.uint8)
    cv2.line(test_img_L, (10, 10), (10, 38), (255), thickness=4)  # Vertical
    cv2.line(test_img_L, (10, 38), (35, 38), (255), thickness=4)  # Horizontal

    # cv2.imshow("Original L for Stroke Perturbation", test_img_L); cv2.waitKey(1)

    print("\n--- Testing Stroke-Based Perturbation ---")
    # 1. Extract strokes
    strokes_from_L = extractor.extract(test_img_L.copy())
    if not strokes_from_L:
        print("Stroke extraction failed for 'L' image. Cannot test perturber.")
    else:
        print(f"Extracted {len(strokes_from_L)} strokes from 'L'.")
        # 2. Perturb image via strokes
        perturbed_L_image = perturber.perturb_image_via_strokes(test_img_L.copy(), strokes_from_L)

        print(f"Original L shape: {test_img_L.shape}, Perturbed L shape: {perturbed_L_image.shape}")
        assert perturbed_L_image.shape == test_img_L.shape
        # Check if image changed (it should if strokes were perturbed)
        # This might fail if random changes are very small or cancel out
        # assert not np.array_equal(test_img_L, perturbed_L_image)

        # Visualize
        # combined_L = np.hstack((test_img_L, perturbed_L_image))
        # cv2.imshow("Original L vs. Stroke Perturbed L", combined_L)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Stroke perturbation test completed (visual check recommended).")