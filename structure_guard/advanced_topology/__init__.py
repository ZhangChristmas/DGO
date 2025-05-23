# ultimate_morph_generator/structure_guard/__init__.py

# 确保这是文件的最开始部分 (或在 future imports 之后)
from __future__ import annotations  # 推荐用于现代类型提示，尤其是有前向引用时

from typing import Optional, List, Any

# 从 config 模块导入所需的配置类
# 这是关键的导入，确保它在任何使用这些类型的地方之前
from ...config import (
    get_config,
    StructureGuardConfig,
    AdvancedTopologyConfig,  # <--- 必须在这里导入
    BasicTopologyConfig  # 如果 BasicTopologyChecker 的类型提示也需要
)
from ...utilities.type_definitions import CvImage
from ...perturbation_suite.stroke_engine.stroke_extractor import Stroke
import logging  # 标准的 logging 获取方式

# 获取 logger
logger = logging.getLogger(__name__)

# 现在可以导入本包内的其他模块
from ..basic_topology import BasicTopologyChecker
from ..advanced_topology.persistent_homology import PersistentHomologyAnalyzer, PersistenceSignature
from ..advanced_topology.graph_representation import CharacterGraphAnalyzer


# StructureGuard 类定义...
class StructureGuard:
    """
    协调各种结构和拓扑检查，以确保
    扰动后的字符保持其基本形态。
    """

    def __init__(self, guard_cfg: StructureGuardConfig,  # 类型提示使用已导入的 StructureGuardConfig
                 target_char_string: str,
                 reference_ph_signature: Optional[PersistenceSignature] = None,
                 ):
        self.cfg = guard_cfg
        self.target_char_string = target_char_string

        self.basic_checker: Optional[BasicTopologyChecker] = None
        # 检查 basic_topology 是否存在且已启用
        if hasattr(self.cfg, 'basic_topology') and self.cfg.basic_topology and self.cfg.basic_topology.enabled:
            self.basic_checker = BasicTopologyChecker(
                topology_cfg=self.cfg.basic_topology,  # 类型提示 BasicTopologyConfig
                target_char_string=self.target_char_string
            )
            logger.info("BasicTopologyChecker 已初始化。")
        else:
            logger.info("基础拓扑检查器在配置中被禁用或未配置。")

        self.ph_analyzer: Optional[PersistentHomologyAnalyzer] = None
        # 检查 advanced_topology 和 persistent_homology_params 是否存在且已启用
        if hasattr(self.cfg, 'advanced_topology') and self.cfg.advanced_topology and \
                self.cfg.advanced_topology.enabled and \
                hasattr(self.cfg.advanced_topology, 'persistent_homology_params') and \
                self.cfg.advanced_topology.persistent_homology_params and \
                self.cfg.advanced_topology.persistent_homology_params.get("enabled", False):

            self.ph_analyzer = PersistentHomologyAnalyzer(
                adv_topology_cfg=self.cfg.advanced_topology  # 类型提示 AdvancedTopologyConfig
            )
            self.reference_ph_signature = reference_ph_signature
            if self.reference_ph_signature:
                logger.info("PersistentHomologyAnalyzer 已初始化并带有参考签名。")
            else:
                logger.warning("PersistentHomologyAnalyzer 已初始化，但未提供参考PH签名。PH检查可能仅限于原始签名生成。")
        else:
            logger.info("持久同调子模块在配置中被禁用或未正确配置。")

        self.graph_analyzer: Optional[CharacterGraphAnalyzer] = None
        # 检查 advanced_topology 和 graph_representation_params 是否存在且已启用
        if hasattr(self.cfg, 'advanced_topology') and self.cfg.advanced_topology and \
                self.cfg.advanced_topology.enabled and \
                hasattr(self.cfg.advanced_topology, 'graph_representation_params') and \
                self.cfg.advanced_topology.graph_representation_params and \
                self.cfg.advanced_topology.graph_representation_params.get("enabled", False):

            self.graph_analyzer = CharacterGraphAnalyzer(
                adv_topology_cfg=self.cfg.advanced_topology,  # 类型提示 AdvancedTopologyConfig
                target_char_string=self.target_char_string
            )
            if self.graph_analyzer.reference_graph:
                logger.info("CharacterGraphAnalyzer 已初始化并带有参考图。")
            else:
                logger.warning("CharacterGraphAnalyzer 已初始化，但没有可用的参考图。图检查将失败或被跳过。")
        else:
            logger.info("图表示子模块在配置中被禁用或未正确配置。")

    # check_image_structure 方法... (保持不变)
    def check_image_structure(self, image: CvImage,
                              extracted_strokes: Optional[List[Stroke]] = None) -> bool:
        logger.debug(f"正在为字符 '{self.target_char_string}' 运行结构检查...")
        # 1. 基础拓扑检查
        if self.basic_checker and self.basic_checker.cfg.enabled:  # 再次检查 basic_checker 是否已初始化
            if not self.basic_checker.run_checks(image.copy()):
                logger.info("结构检查失败: 基础拓扑检查。")
                return False
        else:
            logger.debug("基础拓扑检查器已跳过 (禁用或未初始化)。")
        # ... (PH 和 Graph 检查逻辑保持不变) ...
        if self.ph_analyzer and \
                hasattr(self.cfg.advanced_topology, 'persistent_homology_params') and \
                self.cfg.advanced_topology.persistent_homology_params and \
                self.cfg.advanced_topology.persistent_homology_params.get("enabled", False):  # 确保条件与初始化时一致
            if self.reference_ph_signature:
                if not self.ph_analyzer.compare_to_reference_signature(image.copy(), self.reference_ph_signature):
                    logger.info("结构检查失败: 持久同调签名不匹配。")
                    return False
                logger.debug("持久同调检查通过。")
            else:
                logger.debug("持久同调检查已跳过: 未提供用于比较的参考签名。")
        else:
            logger.debug("持久同调分析器已跳过 (禁用或未初始化)。")

        if self.graph_analyzer and \
                hasattr(self.cfg.advanced_topology, 'graph_representation_params') and \
                self.cfg.advanced_topology.graph_representation_params and \
                self.cfg.advanced_topology.graph_representation_params.get("enabled", False):  # 确保条件与初始化时一致
            if self.graph_analyzer.reference_graph:
                if not self.graph_analyzer.check_structure(image.copy(), strokes=extracted_strokes):
                    logger.info("结构检查失败: 图表示不匹配。")
                    return False
                logger.debug("图表示检查通过。")
            else:
                logger.debug("图表示检查已跳过: 没有可用的参考图进行比较。")
        else:
            logger.debug("图表示分析器已跳过 (禁用或未初始化)。")

        logger.info(f"所有启用的结构检查均已通过，字符为 '{self.target_char_string}'。")
        return True


# --- 用于预计算参考签名的辅助函数 ---
def compute_reference_ph_signature_for_char(char_image_path: str,
                                            adv_topology_cfg: AdvancedTopologyConfig) -> Optional[PersistenceSignature]:
    # 确保导入 cv2 和 os
    import cv2  # 建议移到文件顶部，但如果只在这里用也可以
    import os  # 建议移到文件顶部

    logger.info(f"正在尝试从以下路径计算参考PH签名: {char_image_path}")
    # ... (函数其余部分与之前相同) ...
    if not os.path.exists(char_image_path):
        logger.error(f"参考字符图像未找到: {char_image_path}")
        return None

    ref_image = cv2.imread(char_image_path, cv2.IMREAD_UNCHANGED)
    if ref_image is None:
        logger.error(f"加载参考字符图像失败: {char_image_path}")
        return None

    # 检查 persistent_homology_params 是否存在且已启用
    ph_params_enabled = False
    allow_mock_ph = False
    if hasattr(adv_topology_cfg, 'persistent_homology_params') and \
            adv_topology_cfg.persistent_homology_params:
        ph_params_enabled = adv_topology_cfg.persistent_homology_params.get("enabled", False)
        allow_mock_ph = adv_topology_cfg.persistent_homology_params.get('allow_mock_if_gudhi_unavailable', False)

    if not ph_params_enabled:
        logger.info("PH参考计算已跳过，因为PH在配置中被禁用。")
        return None

    ph_analyzer = PersistentHomologyAnalyzer(adv_topology_cfg)

    if not ph_analyzer.gudhi_available and not allow_mock_ph:
        logger.warning("无法计算参考PH签名: Gudhi 不可用且不允许 mock。")
        return None

    signature = ph_analyzer.get_persistence_signature(ref_image)
    if signature:
        logger.info(f"计算得到的参考PH签名: {signature}")
    else:
        logger.warning("计算参考PH签名失败。")
    return signature

# __main__ block for testing this file
# (保持注释掉或删除，因为它依赖于特定的测试设置和路径)