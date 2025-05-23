# ultimate_morph_generator/config.py
import os

from pydantic import BaseModel, Field, DirectoryPath, FilePath
from typing import List, Tuple, Dict, Optional, Literal, Union
import torch
import yaml

# --- Helper Enums and Literals ---
DeviceLiteral = Literal["cpu", "cuda", "mps", "auto"]
OptimizerLiteral = Literal["Adam", "AdamW", "SGD", "RMSprop"]
LossFunctionLiteral = Literal["CrossEntropyLoss", "NLLLoss"]  # Add more as needed
DGOModelArchitectureLiteral = Literal["BaseCNN", "ResNetVariant", "ViTSmall"]  # Add more as needed
UncertaintyMethodLiteral = Literal["none", "mc_dropout", "ensemble"]  # Add more as needed
ContinualLearningStrategyLiteral = Literal["none", "ewc", "si", "agem", "lwf"]  # Add more as needed
HashingMethodLiteral = Literal[
    "simhash_projection", "thresholding", "multi_scale_concat", "learned_hash"]  # Add more as needed
DiversityStrategyLiteral = Literal["novelty_only", "explore_low_density", "coverage_maximization"]  # Add more as needed
ParentSelectionStrategyLiteral = Literal[
    "random", "high_confidence", "boundary_cases", "low_density_feature", "high_uncertainty"]  # Add more as needed
PerturbationParamSelectionStrategyLiteral = Literal[
    "random", "rl_guided", "ea_guided", "adaptive_bayesian"]  # Add more as needed
LoggingLevelLiteral = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


# --- Sub-configurations ---

class PerturbationMethodConfig(BaseModel):
    enabled: bool = True
    probability_of_application: float = Field(0.5, ge=0.0, le=1.0)  # 此方法被选中的概率
    # 具体参数范围，用Dict[str, Union[List, Tuple]] 表示，例如:
    # params: Dict[str, Union[Tuple[float, float], Tuple[int, int], List[Any]]] = {}
    # 为了更具体，我们为每个方法定义其参数
    # 示例: local_pixel_params, elastic_params etc.
    # 这里用一个通用字典，具体key在实现时定义和检查
    param_ranges: Dict[str, Union[List, Tuple]] = Field(default_factory=dict)


class StrokeEngineConfig(BaseModel):
    enabled: bool = False
    extractor_type: Literal["skeletonization_vectorization", "deep_learning_model"] = "skeletonization_vectorization"
    deep_learning_model_path: Optional[FilePath] = None  # 如果使用深度学习提取器
    min_stroke_length: int = 5  # 最小笔画长度（像素）
    perturbation_probability: float = Field(0.3, ge=0.0, le=1.0)


class StyleMixerConfig(BaseModel):
    enabled: bool = False
    style_source_dir: Optional[DirectoryPath] = None  # 风格图像来源
    strength_range: Tuple[float, float] = (0.05, 0.2)  # 风格混合强度


class PerturbationSuiteConfig(BaseModel):
    max_attempts_per_parent: int = Field(30, ge=1)
    param_selection_strategy: PerturbationParamSelectionStrategyLiteral = "random"
    # 新增字段示例 (确保在 Pydantic 模型中定义它们)
    max_perturb_sequence_len: int = Field(default=3, ge=1)
    dgo_guided_perturb_probability: float = Field(default=0.1, ge=0.0, le=1.0)

    local_pixel: PerturbationMethodConfig = Field(default_factory=lambda: PerturbationMethodConfig(
        enabled=True,
        probability_of_application=0.3,
        param_ranges={
            "neighborhood_size": [3, 5], # 从列表中选择一个
            "perturb_density": (0.01, 0.05)  # 从 (min, max) 均匀随机选择一个浮点数
            # "intensity_noise_range": (-20, 20) # 如果要用这个参数，确保它被 base_transforms.py 处理
        }
    ))
    elastic_deformation: PerturbationMethodConfig = Field(default_factory=lambda: PerturbationMethodConfig(
        enabled=True,
        probability_of_application=0.5,
        param_ranges={
            "alpha": (20.0, 60.0),
            "sigma": (3.0, 7.0),
            "alpha_affine": (0.0, 15.0)
        }
    ))
    fine_affine: PerturbationMethodConfig = Field(default_factory=lambda: PerturbationMethodConfig(
        enabled=True,
        probability_of_application=0.5,
        param_ranges={
            "max_rotation_degrees": (1.0, 10.0),      # _get_params_from_config 会从中随机选一个值
            "max_scale_delta": (0.05, 0.15),
            "max_shear_degrees_x": (1.0, 8.0),
            "max_shear_degrees_y": (1.0, 8.0),
            "translate_percent_x": (-0.05, 0.05),
            "translate_percent_y": (-0.05, 0.05),
        }
    ))
    stroke_thickness_morph: PerturbationMethodConfig = Field(default_factory=lambda: PerturbationMethodConfig(
        enabled=True,
        probability_of_application=0.2,
        param_ranges={
            "operation_type": ["dilate", "erode"], # 从列表中选择一个
            "kernel_size": [3, 5]                 # 从列表中选择一个
        }
    ))
    stroke_engine_perturbations: StrokeEngineConfig = Field(default_factory=StrokeEngineConfig)
    style_mixer: StyleMixerConfig = Field(default_factory=StyleMixerConfig)


class DGOTrainingConfig(BaseModel):
    batch_size: int = Field(64, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    epochs_initial_training: int = Field(10, ge=0)  # 初始训练轮数
    epochs_finetuning: int = Field(3, ge=0)  # 微调轮数
    optimizer: OptimizerLiteral = "AdamW"
    loss_function: LossFunctionLiteral = "CrossEntropyLoss"
    weight_decay: float = Field(1e-4, ge=0)
    scheduler_step_size: Optional[int] = None  # for StepLR scheduler
    scheduler_gamma: Optional[float] = None  # for StepLR scheduler

    use_self_supervised_pretraining: bool = False  # (高级功能占位)
    continual_learning_strategy: ContinualLearningStrategyLiteral = "ewc"
    ewc_lambda: float = Field(100.0, ge=0)  # EWC 正则化强度 (如果使用EWC)
    si_lambda: float = Field(1.0, ge=0)  # SI 正则化强度
    agem_buffer_size: int = Field(100, ge=0)  # A-GEM buffer size


class DGOOracleConfig(BaseModel):
    architecture: DGOModelArchitectureLiteral = "ResNetVariant"
    num_classes: int = Field(10, ge=2)
    pretrained_model_path: Optional[FilePath] = None
    feature_extraction_layer_name: Optional[str] = "avgpool"  # 模型中用于提取特征的层名, e.g., 'avgpool', 'fc1'
    # feature_dim 会自动推断, 但可以作为验证
    # DGO output a classification and a feature vector.
    uncertainty_method: UncertaintyMethodLiteral = "mc_dropout"
    mc_dropout_samples: int = Field(10, ge=1) if uncertainty_method == "mc_dropout" else 1
    training_params: DGOTrainingConfig = Field(default_factory=DGOTrainingConfig)


class BasicTopologyConfig(BaseModel):
    enabled: bool = True
    # 字符 spécifiques 规则, e.g. "3": {"expected_holes": 2, "min_hole_area": 10}
    rules_for_char: Dict[str, Dict[str, Union[int, float, List[str]]]] = Field(default_factory=lambda: {
        "3": {"expected_holes": 2, "min_hole_area": 10, "char_threshold": 80, "opening_directions": ["right", "right"]},
        # 开口方向示例
        "8": {"expected_holes": 2, "min_hole_area": 5, "char_threshold": 80}
    })


class AdvancedTopologyConfig(BaseModel):
    enabled: bool = False
    persistent_homology_params: Dict = Field(default_factory=dict)  # e.g. { "max_edge_length": 0.1 }
    graph_representation_params: Dict = Field(default_factory=dict)


class StructureGuardConfig(BaseModel):
    basic_topology: BasicTopologyConfig = Field(default_factory=BasicTopologyConfig)
    advanced_topology: AdvancedTopologyConfig = Field(default_factory=AdvancedTopologyConfig)


class FeatureAnalysisConfig(BaseModel):
    hashing_method: HashingMethodLiteral = "simhash_projection"
    hash_length: int = Field(128, ge=16)
    novelty_hamming_distance_threshold_ratio: float = Field(0.15, ge=0.0, le=1.0)  # 汉明距离阈值占哈希长度的比例
    diversity_strategy: DiversityStrategyLiteral = "explore_low_density"
    feature_space_density_estimator: Literal["kde", "grid_count"] = "kde"  # 如果 explore_low_density
    kde_bandwidth: float = Field(0.5, gt=0)


class GenerationManagerConfig(BaseModel):
    max_library_size: int = Field(1000, ge=10)
    max_generations_without_improvement: int = Field(200, ge=10)  # "improvement" = new sample added
    parent_selection_strategy: ParentSelectionStrategyLiteral = "boundary_cases"
    # 置信度阈值
    dgo_acceptance_confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)  # 入库的最低置信度
    dgo_boundary_case_confidence_range: Tuple[float, float] = (0.4, 0.8)  # 边缘案例的置信度范围
    dgo_high_confidence_threshold: float = Field(0.95, ge=0.0, le=1.0)  # 用于父本选择或判断扰动效果
    # DGO 微调触发
    dgo_finetune_trigger_new_samples: int = Field(50, ge=1)  # 每生成N个新样本后触发
    dgo_finetune_data_buffer_size: int = Field(200, ge=1)  # 存储用于微调的(图像,标签)对的最大数量
    # 人工反馈模拟 (高级功能占位)
    simulated_human_feedback_interval: Optional[int] = None  # 每N个样本模拟一次人工反馈
    simulated_human_feedback_batch_size: int = Field(10, ge=1)


class DataManagementConfig(BaseModel):
    initial_samples_path_template: str = "./initial_data_pool/char_{char_string}/"
    output_base_dir: DirectoryPath = Field("./generated_morphologies/")
    database_filename: str = "morphology_library.sqlite"
    image_archive_subfolder: str = "image_files"
    image_file_format: Literal[".png", ".jpg", ".bmp"] = ".png"
    target_image_size: Tuple[int, int] = (28, 28)  # H, W
    grayscale_input: bool = True


class LoggingConfig(BaseModel):
    level: LoggingLevelLiteral = "INFO"
    log_to_file: bool = True
    log_file_path: str = "generation_run.log"  # Will be in output_base_dir/reports
    log_to_console: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class SystemConfig(BaseModel):
    project_name: str = "DGO"
    version: str = "1.0.0"
    target_character_index: int = Field(3, ge=0)
    target_character_string: str = "3"
    confusable_character_indices: List[int] = Field(default_factory=lambda: [2, 8])

    random_seed: Optional[int] = 42
    device: DeviceLiteral = "auto"

    data_management: DataManagementConfig = Field(default_factory=DataManagementConfig)
    dgo_oracle: DGOOracleConfig = Field(default_factory=DGOOracleConfig)
    perturbation_suite: PerturbationSuiteConfig = Field(default_factory=PerturbationSuiteConfig)
    structure_guard: StructureGuardConfig = Field(default_factory=StructureGuardConfig)
    feature_analysis: FeatureAnalysisConfig = Field(default_factory=FeatureAnalysisConfig)
    generation_manager: GenerationManagerConfig = Field(default_factory=GenerationManagerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def get_actual_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)

    @classmethod
    def from_yaml(cls, file_path: str) -> "SystemConfig":
        """从 YAML 文件加载配置。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件 YAML 未找到: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"YAML 文件为空或无效: {file_path}")
        return cls.model_validate(data)

    def to_yaml(self, file_path: str):
        """将当前配置保存到 YAML 文件。"""
        config_dict = self.model_dump(mode='json', exclude_none=False)

        dir_name = os.path.dirname(file_path)
        if dir_name:  # 仅当 file_path 包含目录路径时才创建目录
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, sort_keys=False, indent=2, allow_unicode=True)
        print(f"配置已保存至 YAML: {file_path}")


# --- Main function to load or create default config ---
_config_instance: Optional[SystemConfig] = None


def get_config(config_file_path: Optional[str] = None, reload: bool = False) -> SystemConfig:  # Added reload flag
    """
    Loads the system configuration from a YAML file or returns/creates a default.
    Singleton pattern for config instance unless reload is True.
    """
    global _config_instance
    if _config_instance is None or reload:  # Load if not loaded or if reload is forced
        if config_file_path and os.path.exists(config_file_path):
            try:
                print(f"Loading configuration from YAML: {config_file_path}")
                _config_instance = SystemConfig.from_yaml(config_file_path)
            except FileNotFoundError:  # Should be caught by from_yaml but as fallback
                print(f"Warning: Config file {config_file_path} not found. Using default config.")
                _config_instance = SystemConfig()
            except Exception as e:
                print(f"Error loading configuration from {config_file_path}: {e}. Using default config.")
                _config_instance = SystemConfig()
        else:
            if config_file_path:  # Path given but does not exist
                print(
                    f"Warning: Config file path '{config_file_path}' provided but file not found. Using default config.")
            else:  # No path given
                print("No configuration file path provided. Using default config.")
            _config_instance = SystemConfig()
    return _config_instance


if __name__ == "__main__":
    # --- Example of using the YAML loading/saving ---

    # 1. Create and save a default configuration to YAML
    default_cfg_instance = get_config(reload=True)  # Get a fresh default instance
    default_yaml_path = "default_generated_config.yaml"
    print(f"\nSaving default configuration to: {default_yaml_path}")
    default_cfg_instance.to_yaml(default_yaml_path)

    # 2. Load configuration from the saved YAML file
    print(f"\nLoading configuration from: {default_yaml_path}")
    loaded_cfg_from_yaml = get_config(config_file_path=default_yaml_path, reload=True)  # Force reload from file

    # Verify some loaded values
    print(f"Loaded Project Name: {loaded_cfg_from_yaml.project_name}")
    print(f"Loaded DGO Model: {loaded_cfg_from_yaml.dgo_oracle.model_architecture}")
    assert loaded_cfg_from_yaml.project_name == default_cfg_instance.project_name  # Simple check

    # 3. Example of trying to load a non-existent file (should use defaults)
    print("\nAttempting to load non-existent config (should use defaults):")
    cfg_non_existent = get_config(config_file_path="non_existent_config.yaml", reload=True)
    print(f"Project Name from non-existent load: {cfg_non_existent.project_name} (should be default)")
    assert cfg_non_existent.project_name == "DGO"  # Assuming this is the default

    # 4. Show how get_config acts as a singleton (without reload=True)
    print("\nTesting singleton behavior of get_config():")
    first_call_cfg = get_config(reload=True)  # Ensure fresh default
    first_call_cfg.project_name = "ModifiedInFirstCall"  # Modify the instance

    second_call_cfg = get_config()  # No path, no reload -> should return the modified instance
    print(f"Project name from second call: {second_call_cfg.project_name}")
    assert second_call_cfg.project_name == "ModifiedInFirstCall"

    # Clean up the generated default YAML file
    # if os.path.exists(default_yaml_path):
    #     os.remove(default_yaml_path)
    #     print(f"\nCleaned up {default_yaml_path}")