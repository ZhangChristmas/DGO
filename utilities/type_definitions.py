# ultimate_morph_generator/utilities/type_definitions.py
import numpy as np
import torch
from typing import NewType, Tuple, List, Dict, Any, Union
from PIL.Image import Image as PILImage  # type: ignore

# --- Primitive Types with Semantic Meaning ---
ImagePath = NewType('ImagePath', str)
ImageHash = NewType('ImageHash', np.ndarray)  # 通常是二值向量 (uint8)
FeatureVector = NewType('FeatureVector', np.ndarray)  # 通常是浮点向量 (float32)

# --- Image Data Types ---
# OpenCV图像通常是 (H, W, C) or (H, W) NumPy arrays, BGR order for color
CvImage = np.ndarray  # np.uint8
# PyTorch张量通常是 (B, C, H, W) or (C, H, W)
TorchImageTensor = torch.Tensor  # Typically float32, normalized

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
# This could be a Pydantic model or a complex tuple/dict
class MorphologySample(BaseModel):  # 引入Pydantic BaseModel以便于序列化
    sample_id: str  # Unique ID, e.g., UUID or auto-increment from DB
    image_path: ImagePath
    image_hash: List[int]  # 哈希码存储为Python列表 (0或1)
    feature_vector_path: Optional[str] = None  # 特征向量可以很大，选择性存储为文件

    dgo_predicted_label: int
    dgo_confidence: float
    dgo_uncertainty: Optional[float] = None

    generation_step: int
    parent_id: Optional[str] = None  # 父样本的ID
    perturbation_applied: Optional[PerturbationName] = None
    perturbation_params_applied: Optional[PerturbationParams] = None  # 存储实际应用的参数

    structure_check_passed: bool
    novelty_score: Optional[float] = None  # e.g., min hamming distance to existing samples

    creation_timestamp: float  # time.time()

    # 更多元数据...
    # human_label: Optional[int] = None
    # quality_rating: Optional[float] = None

    def get_image_hash_np(self) -> ImageHash:
        return np.array(self.image_hash, dtype=np.uint8)

    class Config:
        arbitrary_types_allowed = True  # 允许 np.ndarray 等类型，虽然这里都转为Python原生了


# --- For function signatures and clarity ---
Label = int  # 通常是类别索引

# 可以根据需要添加更多自定义类型

if __name__ == "__main__":
    # 示例用法
    dummy_hash_list = [0, 1, 0, 1, 1, 0]
    dummy_hash_np: ImageHash = np.array(dummy_hash_list, dtype=np.uint8)

    print(f"ImageHash (np.ndarray): {dummy_hash_np}")

    sample_entry = MorphologySample(
        sample_id="sample_001",
        image_path=ImagePath("path/to/image.png"),
        image_hash=dummy_hash_list,
        dgo_predicted_label=3,
        dgo_confidence=0.98,
        generation_step=10,
        structure_check_passed=True,
        creation_timestamp=time.time()
    )
    print(f"\nMorphologySample instance:\n{sample_entry.model_dump_json(indent=2)}")
    print(f"Sample hash (numpy): {sample_entry.get_image_hash_np()}")