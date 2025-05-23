# ultimate_morph_generator/core_logic/stopping_conditions.py
from typing import Optional, Dict, Any

from ..config import get_config, GenerationManagerConfig
from ..dgo_oracle.dgo_model_handler import DGOModelHandler  # If checking DGO performance
from ..utilities.logging_config import setup_logging

logger = setup_logging()


class StoppingConditionChecker:
    """
    Encapsulates various stopping conditions for the generation process.
    """

    def __init__(self, gen_cfg: GenerationManagerConfig,
                 # Other components might be needed for specific conditions
                 dgo_handler: Optional[DGOModelHandler] = None,
                 # validation_dataloader: Optional[DataLoader] = None # For DGO perf check
                 ):
        self.gen_cfg = gen_cfg
        self.dgo_handler = dgo_handler
        # self.validation_dataloader = validation_dataloader

        # Thresholds from config (can be extended)
        self.target_dgo_val_accuracy: Optional[float] = getattr(self.gen_cfg, 'target_dgo_validation_accuracy', None)
        self.min_dgo_val_accuracy_drop_streak: Optional[int] = getattr(self.gen_cfg,
                                                                       'min_dgo_val_accuracy_drop_streak_for_stop',
                                                                       None)

        self._dgo_val_accuracy_history: List[float] = []
        self._accuracy_drop_streak_count: int = 0

    def check_library_size(self, current_library_size: int) -> bool:
        """Checks if the maximum library size has been reached."""
        if current_library_size >= self.gen_cfg.max_library_size:
            logger.info(f"Stopping condition met: Max library size ({self.gen_cfg.max_library_size}) reached.")
            return True
        return False

    def check_no_improvement_streak(self, consecutive_no_new_samples: int) -> bool:
        """Checks if the maximum number of generations without new samples has been reached."""
        if consecutive_no_new_samples >= self.gen_cfg.max_generations_without_improvement:
            logger.info(
                f"Stopping condition met: Max generations without new sample ({self.gen_cfg.max_generations_without_improvement}) reached.")
            return True
        return False

    def check_dgo_validation_performance(self) -> bool:
        """
        Checks DGO performance on a fixed validation set.
        Stops if accuracy reaches a target, or consistently drops.
        (This is a conceptual placeholder requiring a validation dataloader and evaluation logic).
        """
        if not self.dgo_handler or not getattr(self.gen_cfg, 'enable_dgo_validation_stopping',
                                               False):  # Add to GenManagerConfig
            return False  # Condition disabled or DGO not available

        # --- Conceptual: Evaluate DGO on a validation set ---
        # current_val_accuracy = self.dgo_handler.evaluate_accuracy(self.validation_dataloader)
        # if current_val_accuracy is None: return False
        # logger.info(f"DGO Validation Accuracy: {current_val_accuracy:.4f}")
        # self._dgo_val_accuracy_history.append(current_val_accuracy)

        # if self.target_dgo_val_accuracy and current_val_accuracy >= self.target_dgo_val_accuracy:
        #     logger.info(f"Stopping condition met: DGO validation accuracy target ({self.target_dgo_val_accuracy:.3f}) reached.")
        #     return True

        # if self.min_dgo_val_accuracy_drop_streak and len(self._dgo_val_accuracy_history) > 1:
        #     if current_val_accuracy < self._dgo_val_accuracy_history[-2]: # Accuracy dropped
        #         self._accuracy_drop_streak_count += 1
        #     else: # Accuracy did not drop
        #         self._accuracy_drop_streak_count = 0

        #     if self._accuracy_drop_streak_count >= self.min_dgo_val_accuracy_drop_streak:
        #         logger.info(f"Stopping condition met: DGO validation accuracy dropped for {self._accuracy_drop_streak_count} consecutive evaluations.")
        #         return True
        # --- End Conceptual ---

        logger.debug("DGO validation performance check is conceptual. Not stopping based on it yet.")
        return False

    def check_all(self, current_library_size: int, consecutive_no_new_samples: int) -> bool:
        """Checks all configured stopping conditions."""
        if self.check_library_size(current_library_size): return True
        if self.check_no_improvement_streak(consecutive_no_new_samples): return True
        if self.check_dgo_validation_performance(): return True  # Conceptual
        # Add more conditions here
        return False


if __name__ == "__main__":
    # --- Test StoppingConditionChecker ---
    from ....config import SystemConfig  # Adjust relative import

    temp_sys_cfg_data_stop = {
        "generation_manager": {
            "max_library_size": 100,
            "max_generations_without_improvement": 10,
            # "enable_dgo_validation_stopping": True, # For conceptual test
            # "target_dgo_validation_accuracy": 0.95,
            # "min_dgo_val_accuracy_drop_streak_for_stop": 3
        },
        "logging": {"level": "DEBUG"}
    }
    _config_instance = SystemConfig.model_validate(temp_sys_cfg_data_stop)
    cfg_glob_stop = get_config()

    checker = StoppingConditionChecker(gen_cfg=cfg_glob_stop.generation_manager)

    print("\n--- Testing Stopping Conditions ---")
    print(f"Lib size 50 (<100), No improvement 5 (<10): Stop? {checker.check_all(50, 5)}")
    assert not checker.check_all(50, 5)

    print(f"Lib size 100 (>=100), No improvement 5 (<10): Stop? {checker.check_all(100, 5)}")
    assert checker.check_all(100, 5)

    print(f"Lib size 50 (<100), No improvement 10 (>=10): Stop? {checker.check_all(50, 10)}")
    assert checker.check_all(50, 10)

    # Conceptual DGO validation test (if logic were implemented)
    # checker.dgo_handler = DGOModelHandler(...) # Needs a dummy DGO
    # checker.validation_dataloader = DataLoader(...) # Needs a dummy DataLoader
    # checker._dgo_val_accuracy_history = [0.8, 0.7, 0.6] # Simulate drops
    # checker._accuracy_drop_streak_count = 3
    # print(f"Simulated DGO val acc drop: Stop? {checker.check_dgo_validation_performance()}")
    # assert checker.check_dgo_validation_performance() if cfg_glob_stop.generation_manager.enable_dgo_validation_stopping else True

    print("\nStoppingConditionChecker tests completed.")