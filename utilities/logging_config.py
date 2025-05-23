# ultimate_morph_generator/utilities/logging_config.py
import logging
import os
import sys
from ..config import LoggingConfig, get_config  # 使用相对导入


def setup_logging():
    """
    Configures the logging for the entire application based on SystemConfig.
    Returns the main logger instance.
    """
    cfg = get_config()
    log_cfg = cfg.logging

    # 创建主日志记录器
    logger = logging.getLogger(cfg.project_name)  # 使用项目名称作为根日志记录器名称
    logger.setLevel(log_cfg.level.upper())  # 设置日志级别

    # 移除已存在的handlers，避免重复添加 (尤其在Jupyter环境中)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(log_cfg.log_format)

    handlers = []

    if log_cfg.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_cfg.log_to_file:
        # 确保日志文件路径存在
        # 日志文件放在 generated_output/reports/ 目录下
        reports_dir = os.path.join(cfg.data_management.output_base_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        file_path = os.path.join(reports_dir, log_cfg.log_file_path)

        # 使用 RotatingFileHandler 可以限制日志文件大小并创建备份
        # from logging.handlers import RotatingFileHandler
        # file_handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')

        # 简单文件处理器
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')  # 'a' for append
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if not handlers:  # 如果既不输出到控制台也不输出到文件，则添加一个NullHandler避免"No handlers could be found"警告
        logger.addHandler(logging.NullHandler())
    else:
        for handler in handlers:
            logger.addHandler(handler)

    # 可以为特定库（如torch, matplotlib）设置更高级别的日志，以减少冗余信息
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info(
        f"Logging initialized. Level: {log_cfg.level}. Output to console: {log_cfg.log_to_console}, file: {log_cfg.log_to_file if log_cfg.log_to_file else 'None'}.")
    return logger


# 在模块加载时自动配置日志是一个常见的做法，但有时可能太早。
# 更好的方式是在主程序入口显式调用 setup_logging()。
# logger = setup_logging() # 或者在main_orchestrator.py中调用

if __name__ == "__main__":
    # 这个main仅用于测试这个模块
    # 为了测试，需要一个临时的config实例
    from ..config import SystemConfig

    # 创建一个临时的配置实例来测试日志
    # 在实际应用中，get_config()会处理配置的加载
    # temp_cfg_data = {
    #     "project_name": "LoggingTest",
    #     "logging": {
    #         "level": "DEBUG",
    #         "log_to_file": True,
    #         "log_file_path": "test_run.log", # 相对路径，将在当前目录生成
    #         "log_to_console": True
    #     },
    #     "data_management": { # 需要提供 output_base_dir
    #         "output_base_dir": "./temp_generated_output/"
    #     }
    # }
    # # 更新全局配置实例以进行测试
    # from ..config import _config_instance
    # _config_instance = SystemConfig.model_validate(temp_cfg_data)

    # 或者更简单地，让get_config用默认值
    if not os.path.exists("./temp_generated_output/"):
        os.makedirs("./temp_generated_output/reports", exist_ok=True)

    logger_instance = setup_logging()
    logger_instance.debug("This is a debug message.")
    logger_instance.info("This is an info message.")
    logger_instance.warning("This is a warning message.")
    logger_instance.error("This is an error message.")
    logger_instance.critical("This is a critical message.")

    # 检查日志文件是否生成在 temp_generated_output/reports/test_run.log
    # (如果log_file_path是相对的，则在reports_dir下)
    print(
        f"Check log file at: {os.path.join(get_config().data_management.output_base_dir, 'reports', get_config().logging.log_file_path)}")