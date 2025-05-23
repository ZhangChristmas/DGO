# ultimate_morph_generator/perturbation_suite/stroke_engine/stroke_extractor.py
import cv2
import numpy as np
from skimage.morphology import skeletonize  # type: ignore # For skeletonization
from skimage.util import invert  # type: ignore
from typing import List, Tuple, Dict, Any, Optional

from ...config import get_config, StrokeEngineConfig
from ...utilities.type_definitions import CvImage
from ...utilities.logging_config import setup_logging

logger = setup_logging()


# --- Data structure for a single stroke ---
class Stroke:
    """Represents a single extracted stroke, typically as a sequence of points."""

    def __init__(self, stroke_id: int, points: np.ndarray,
                 avg_thickness: Optional[float] = None,
                 start_point_type: Optional[str] = None,  # e.g., 'endpoint', 'junction'
                 end_point_type: Optional[str] = None):
        self.id = stroke_id
        self.points = points  # Nx2 array of (x, y) coordinates
        self.avg_thickness = avg_thickness
        self.start_point_type = start_point_type
        self.end_point_type = end_point_type
        # Could add more attributes: length, curvature, average_intensity, etc.

    def __repr__(self) -> str:
        return f"Stroke(id={self.id}, num_points={len(self.points)}, thickness={self.avg_thickness:.2f if self.avg_thickness else 'N/A'})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "points": self.points.tolist(),  # For JSON serialization
            "avg_thickness": self.avg_thickness,
            "start_point_type": self.start_point_type,
            "end_point_type": self.end_point_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stroke":
        return cls(
            stroke_id=data["id"],
            points=np.array(data["points"]),
            avg_thickness=data.get("avg_thickness"),
            start_point_type=data.get("start_point_type"),
            end_point_type=data.get("end_point_type")
        )


class StrokeExtractor:
    """
    Extracts stroke primitives from a character image.
    """

    def __init__(self, stroke_cfg: StrokeEngineConfig):
        self.cfg = stroke_cfg
        # Threshold for binarizing the image if not already binary
        self.binarization_threshold = getattr(self.cfg, 'binarization_threshold', 128)  # Add to config if needed
        self.min_stroke_pixel_length = self.cfg.min_stroke_length  # Min length in pixels for a polyline segment to be a stroke

    def _preprocess_for_skeletonization(self, image: CvImage) -> CvImage:
        """Prepares the image for skeletonization."""
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        # Binarize: Character should be white (True/1/255) on black background (False/0) for skimage.skeletonize
        # If input is dark char on light bg, invert it first.
        # Heuristic: if mean > threshold, assume light bg.
        if np.mean(gray_image) > self.binarization_threshold:  # Light background
            binary_image = cv2.threshold(gray_image, self.binarization_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        else:  # Dark background
            binary_image = cv2.threshold(gray_image, self.binarization_threshold, 255, cv2.THRESH_BINARY)[1]

        # Normalize to 0 or 1 for skeletonize (skimage expects boolean or 0/1)
        return (binary_image / 255).astype(bool)

    def _vectorize_skeleton(self, skeleton_img: CvImage) -> List[np.ndarray]:
        """
        Converts a skeleton image into a list of polylines (strokes).
        This is a complex step. A simple approach:
        1. Find contours on the skeleton. Each contour is a potential stroke path.
        2. Simplify contours using Douglas-Peucker algorithm (cv2.approxPolyDP).
        3. Break down complex contours at junction points (harder part).

        A more robust method would involve graph traversal of the skeleton pixels,
        identifying endpoints and junction points, and tracing paths between them.

        For this example, we'll use a simplified contour-based approach.
        It might not perfectly separate crossing strokes or handle junctions well.
        """
        if skeleton_img.dtype == bool:
            skeleton_uint8 = (skeleton_img * 255).astype(np.uint8)
        else:
            skeleton_uint8 = skeleton_img.astype(np.uint8)  # Ensure uint8

        contours, hierarchy = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        polylines: List[np.ndarray] = []
        for contour in contours:
            if len(contour) < 2: continue  # Need at least 2 points for a line

            # Simplify contour to get a polyline
            # Epsilon for approxPolyDP: percentage of arc length. Smaller epsilon = more points.
            # A value of 1-2 pixels can be a good starting point for char skeletons.
            epsilon = 0.01 * cv2.arcLength(contour, closed=False)  # If skeleton contours are not closed
            # epsilon = 1.5 # Fixed pixel epsilon
            poly = cv2.approxPolyDP(contour, epsilon, closed=False)  # Skeletons are typically not closed shapes

            # Reshape poly to (N, 2) from (N, 1, 2)
            poly = poly.reshape(-1, 2)

            # Filter by length (pixel length of the polyline)
            # approx_length = cv2.arcLength(poly.reshape(-1,1,2), closed=False) # Recalculate arcLength on simplified
            # A simpler length: sum of segment lengths
            if len(poly) > 1:
                # Simplified length: distance between first and last point (not accurate for curves)
                # Or sum of euclidean distances between consecutive points
                path_length = np.sum(np.sqrt(np.sum(np.diff(poly, axis=0) ** 2, axis=1)))
                if path_length >= self.min_stroke_pixel_length:
                    polylines.append(poly)

        # TODO: Advanced: Split polylines at junction points.
        # This requires identifying pixels with >2 neighbors in the skeleton,
        # and then breaking contours/polylines that pass through these junctions.
        # This is non-trivial with simple contour processing.
        # Graph-based skeleton traversal is better for this.

        return polylines

    def _estimate_stroke_thickness(self, original_binary_image: CvImage, stroke_points: np.ndarray) -> Optional[float]:
        """
        Estimates the average thickness along a stroke path on the original (non-skeletonized) binary image.
        One method: Use Distance Transform on the inverted binary image.
        The values of the distance transform at the skeleton points give half the thickness.
        """
        if original_binary_image.dtype == bool:  # Ensure uint8, 0 for bg, 255 for fg
            original_binary_uint8 = (original_binary_image * 255).astype(np.uint8)
        else:
            original_binary_uint8 = original_binary_image

        # Distance transform on the character pixels (foreground)
        # We want distance from a point on skeleton to the *edge* of the character.
        # So, distance transform on the character itself (not its inverse).
        # Values are distance to nearest background pixel.
        dist_transform = cv2.distanceTransform(original_binary_uint8, cv2.DIST_L2, maskSize=5)

        thicknesses = []
        for pt in stroke_points:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
                # Distance transform value is roughly radius if stroke is circular
                # So, 2 * value is diameter/thickness.
                thicknesses.append(dist_transform[y, x] * 2.0)

        return float(np.mean(thicknesses)) if thicknesses else None

    def extract_strokes_cv(self, image: CvImage) -> List[Stroke]:
        """
        Extracts strokes using traditional Computer Vision techniques (skeletonization).
        Returns a list of Stroke objects.
        """
        logger.debug("Extracting strokes using CV (skeletonization)...")

        # 1. Preprocess and Binarize
        # `binary_char_white_on_black` is True/False or 1/0 for character pixels
        binary_char_white_on_black = self._preprocess_for_skeletonization(image.copy())

        # 2. Skeletonize
        # skimage.skeletonize expects binary image where True means object.
        skeleton = skeletonize(binary_char_white_on_black)

        # For visualization or further CV ops, convert skeleton back to uint8
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        # cv2.imshow("Skeleton", skeleton_uint8); cv2.waitKey(1) # Debug display

        # 3. Vectorize skeleton into polylines
        polylines = self._vectorize_skeleton(skeleton.copy())  # Pass boolean skeleton

        # 4. Create Stroke objects, estimate thickness
        extracted_strokes: List[Stroke] = []
        for i, poly_pts in enumerate(polylines):
            # Thickness estimation needs the original binary image (not skeleton)
            avg_thickness = self._estimate_stroke_thickness(binary_char_white_on_black, poly_pts)

            # TODO: Determine start/end point types (endpoint, junction) by analyzing skeleton neighbors.
            # This is also non-trivial. Requires graph representation or local neighborhood analysis on skeleton.

            stroke = Stroke(stroke_id=i, points=poly_pts, avg_thickness=avg_thickness)
            extracted_strokes.append(stroke)

        logger.info(f"CV Extracted {len(extracted_strokes)} candidate strokes.")
        return extracted_strokes

    def extract_strokes_deep_learning(self, image: CvImage) -> List[Stroke]:
        """
        Placeholder for extracting strokes using a deep learning model.
        This would require a pre-trained model specific to stroke extraction.
        E.g., models that output control points for Bezier curves, or segmentation masks per stroke.
        """
        logger.warning("Deep learning based stroke extraction is a placeholder. Requires a specialized model.")
        # Example workflow:
        # 1. Preprocess image for DL model.
        # 2. Pass through DL model to get parametric strokes (e.g., Bezier control points) or stroke masks.
        # 3. Convert DL model output to List[Stroke] objects.
        # This is highly model-dependent.
        if self.cfg.deep_learning_model_path and os.path.exists(self.cfg.deep_learning_model_path):
            # Load model, run inference...
            pass
        return []  # Return empty list for placeholder

    def extract(self, image: CvImage) -> List[Stroke]:
        """
        Main extraction method, chooses based on configuration.
        """
        if self.cfg.extractor_type == "skeletonization_vectorization":
            return self.extract_strokes_cv(image)
        elif self.cfg.extractor_type == "deep_learning_model":
            return self.extract_strokes_deep_learning(image)
        else:
            logger.error(f"Unknown stroke extractor type: {self.cfg.extractor_type}")
            return []


if __name__ == "__main__":
    # --- Test StrokeExtractor ---
    from ....config import SystemConfig  # Adjust relative import based on test execution context

    # Create a dummy config for testing StrokeEngineConfig
    temp_stroke_engine_cfg_data = {
        "enabled": True,  # This 'enabled' is for the whole engine, not extractor itself
        "extractor_type": "skeletonization_vectorization",
        "min_stroke_length": 5  # pixels
        # "binarization_threshold": 100 # Can add to config
    }
    # stroke_config = StrokeEngineConfig.model_validate(temp_stroke_engine_cfg_data) # This doesn't work directly as it's nested
    # We need to put it inside a dummy SystemConfig structure for get_config() to work if used by StrokeExtractor
    # Or, pass stroke_config directly to StrokeExtractor if its __init__ is refactored.

    # For this test, let's instantiate StrokeEngineConfig directly.
    stroke_config = StrokeEngineConfig(**temp_stroke_engine_cfg_data)

    extractor = StrokeExtractor(stroke_cfg=stroke_config)
    extractor.binarization_threshold = 100  # Set manually for test

    # Create a simple test image (e.g., a letter 'T' or 'L')
    test_img = np.zeros((64, 64), dtype=np.uint8)  # Black background
    # Draw a thick 'T' (white on black)
    cv2.line(test_img, (15, 10), (45, 10), (255), thickness=6)  # Horizontal part
    cv2.line(test_img, (30, 10), (30, 50), (255), thickness=5)  # Vertical part

    # cv2.imshow("Test Char for Stroke Extraction", test_img); cv2.waitKey(1)

    print("\n--- Testing Stroke Extraction (CV method) ---")
    extracted_strokes_list = extractor.extract(test_img.copy())

    if not extracted_strokes_list:
        print("No strokes extracted. Check binarization, skeletonization, or vectorization parameters.")
    else:
        print(f"Extracted {len(extracted_strokes_list)} strokes:")
        for s in extracted_strokes_list:
            print(f"  {s}")
            assert s.points.ndim == 2 and s.points.shape[1] == 2
            if s.avg_thickness is not None:
                assert s.avg_thickness > 0

        # Visualize strokes on the original image (optional)
        vis_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, stroke_obj in enumerate(extracted_strokes_list):
            color = colors[i % len(colors)]
            for pt_idx in range(len(stroke_obj.points) - 1):
                p1 = tuple(stroke_obj.points[pt_idx].astype(int))
                p2 = tuple(stroke_obj.points[pt_idx + 1].astype(int))
                cv2.line(vis_img, p1, p2, color, thickness=1)  # Draw thin lines for extracted paths
                cv2.circle(vis_img, p1, 1, color, -1)  # Draw points
            if len(stroke_obj.points) > 0:
                cv2.circle(vis_img, tuple(stroke_obj.points[-1].astype(int)), 1, color, -1)  # Last point

        # cv2.imshow("Extracted Strokes Overlay", vis_img); cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Stroke extraction test completed (visual check recommended if display enabled).")