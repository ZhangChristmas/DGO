# ultimate_morph_generator/utilities/type_definitions.py
import numpy as np
import torch
from typing import NewType, Tuple, List, Dict, Any, Union, Optional  # <--- 确保 Optional 在这里
from PIL.Image import Image as PILImage  # type: ignore
from pydantic import BaseModel  # <--- 确保 BaseModel 在这里 (因为 MorphologySample 用到了)
import time  # <--- 确保 time 在这里 (因为 MorphologySample 用到了)

# --- Primitive Types with Semantic Meaning ---
ImagePath = NewType('ImagePath', str)
ImageHash = NewType('ImageHash', np.ndarray)  # 通常是二值向量 (uint8)
FeatureVector = NewType('FeatureVector', np.ndarray)  # 通常是浮点向量 (float32)

# --- Image Data Types ---
CvImage = np.ndarray  # np.uint8
TorchImageTensor = torch.Tensor

# --- DGO Related Types ---
DGOScores = np.ndarray  # (num_classes,) probabilities or logits
DGOUncertainty = float  # 单个不确定性度量值
DGOOutput = Tuple[int, float, DGOScores, Optional[FeatureVector], Optional[DGOUncertainty]]
# (predicted_class_idx, confidence, all_class_scores, feature_vector, uncertainty_metric)

# --- Perturbation Related Types ---
PerturbationName = str
PerturbationParams = Dict[str, Any]
PerturbedImageResult = Tuple[CvImage, PerturbationName, PerturbationParams]


# --- Morphology Library Entry ---
class MorphologySample(BaseModel):
    sample_id: str
    image_path: ImagePath
    image_hash: List[int]
    feature_vector_path: Optional[str] = None

    dgo_predicted_label: int
    dgo_confidence: float
    dgo_uncertainty: Optional[float] = None

    generation_step: int
    parent_id: Optional[str] = None
    perturbation_applied: Optional[PerturbationName] = None
    perturbation_params_applied: Optional[PerturbationParams] = None

    structure_check_passed: bool
    novelty_score: Optional[float] = None

    creation_timestamp: float

    def get_image_hash_np(self) -> ImageHash:
        return np.array(self.image_hash, dtype=np.uint8)

    class Config:
        arbitrary_types_allowed = True


# --- For function signatures and clarity ---
Label = int  # 通常是类别索引

if __name__ == "__main__":
    # 示例用法
    dummy_hash_list = [0, 1, 0, 1, 1, 0]
    dummy_hash_np: ImageHash = np.array(dummy_hash_list, dtype=np.uint8)

    print(f"ImageHash (np.ndarray): {dummy_hash_np}")

    sample_entry_data = {  # 使用字典来初始化，因为 ImagePath 不能直接实例化
        "sample_id": "sample_001",
        "image_path": "path/to/image.png",  # Pydantic 会处理 NewType
        "image_hash": dummy_hash_list,
        "dgo_predicted_label": 3,
        "dgo_confidence": 0.98,
        "generation_step": 10,
        "structure_check_passed": True,
        "creation_timestamp": time.time()
    }
    sample_entry = MorphologySample(**sample_entry_data)
    print(f"\nMorphologySample instance:\n{sample_entry.model_dump_json(indent=2)}")
    print(f"Sample hash (numpy): {sample_entry.get_image_hash_np()}")