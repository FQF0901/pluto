import logging
from typing import Optional

import hydra
import numpy
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from omegaconf import DictConfig

from src.custom_training.custom_training_builder import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)

import os
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '$HOME/fqf/nuplan/dataset')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '$HOME/fqf/nuplan/dataset/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '$HOME/fqf/nuplan/dataset/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.1')


logging.getLogger("numba").setLevel(logging.WARNING)    # 将 numba 模块的日志级别设置为 WARNING，以减少不必要的日志输出
logger = logging.getLogger(__name__)    # 获取当前模块的日志记录器实例，用于后续的日志记录

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()  # 要小心确认 ！

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = "./config"
CONFIG_NAME = "default_training"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)   # Hydra库的装饰器@hydra.main：指定配置文件的路径和名称，作为参数传递给主函数
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)  # pytorch_lightning：一个用于简化 PyTorch 模型训练过程并提供高层次的抽象和便捷的功能的库

    # Configure logger, nuplan的logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)   # nuplan库函数

    # Build worker
    worker = build_worker(cfg)  # 4线程并行

    # 修改数据集路径: handcode by fqf
    cfg.scenario_builder.data_root = '/home/fqf/nuplan/dataset/nuplan-v1.1/splits/mini'

    if cfg.py_func == "train":  # 在train_pluto.yaml里配置的
        # Build training engine
        # ProfilerContextManager为nuplan库函数，是一个上下文管理器，用于在特定代码块执行期间进行性能分析（profiling）。它可以帮助开发者记录和分析代码的运行时间、资源使用情况等性能指标
        # ProfilerContextManager 的一般用法是通过 with 语句包裹需要进行性能分析的代码块。其常见参数包括：
        # output_dir：指定性能分析结果的输出目录。
        # enable_profiling：布尔值，决定是否启用性能分析。
        # description：描述标签，用于标识当前性能分析的上下文，方便区分不同部分的性能数据
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"): # 输出目录、是否启用性能分析、描述标签
            engine = build_training_engine(cfg, worker) # 创建训练器、模型和数据模块，并将它们组合成一个 TrainingEngine 对象

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):  # 启用性能分析
            engine.trainer.fit(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )   # planner入口：调用 engine.trainer.fit 方法进行模型训练，传入模型、数据模块和检查点路径
        return engine
    if cfg.py_func == "validate":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
        return engine
    elif cfg.py_func == "test":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info("Starting testing...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == "cache":
        # Precompute and cache all features
        logger.info("Starting caching...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")


if __name__ == "__main__":
    main()
