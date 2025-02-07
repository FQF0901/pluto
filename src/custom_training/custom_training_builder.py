import logging
import os
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import cast

import pytorch_lightning as pl
from hydra.utils import instantiate
from nuplan.planning.script.builders.data_augmentation_builder import (
    build_agent_augmentor,
)
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.objectives_builder import build_objectives
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.splitter_builder import build_splitter
from nuplan.planning.script.builders.training_metrics_builder import (
    build_training_metrics,
)
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from .custom_datamodule import CustomDataModule

logger = logging.getLogger(__name__)


def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions. 这段代码的主要功能是根据某些条件更新配置文件 cfg
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)    # 通过 OmegaConf.set_struct(cfg, False) 使配置文件可编辑

    if cfg.cache.cache_path is None:    # 如果 cache_path 未设置，则禁用缓存并记录警告
        logger.warning("Parameter cache_path is not set, caching is disabled")
    else:
        if not str(cfg.cache.cache_path).startswith("s3://"):   # 如果 cache_path 设置且不是 S3 路径
            if cfg.cache.cleanup_cache and Path(cfg.cache.cache_path).exists():
                rmtree(cfg.cache.cache_path)

            Path(cfg.cache.cache_path).mkdir(parents=True, exist_ok=True)

    if cfg.lightning.trainer.overfitting.enable:    # 如果启用了过拟合模式，则将数据加载器的线程数设为 0。
        cfg.data_loader.params.num_workers = 0

    OmegaConf.resolve(cfg)  # 解析配置中的所有引用和插值，确保配置文件中的所有变量都被正确解析并替换为实际值

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True) # 使配置文件不可编辑。

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:  # 如果启用了日志记录，则记录实验名称、组名及最终配置
        logger.info(
            f"Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config..."
        )
        logger.info("\n" + OmegaConf.to_yaml(cfg))


@dataclass(frozen=True)
class TrainingEngine:
    """
    Lightning 训练引擎数据类，封装了 PyTorch Lightning 的训练器、模型和数据模块。
    该数据类用于将训练所需的组件（包括训练器、模型和数据模块）组合在一起，
    并确保这些组件一旦设置后不可变，从而提高训练过程的稳定性和可预测性。
    """

    trainer: pl.Trainer # 训练器，负责协调训练过程

    model: pl.LightningModule   # 模型模块，描述神经网络模型、损失函数、评估指标和可视化，封装了模型的行为

    datamodule: pl.LightningDataModule  # 数据模块，提供加载和预处理数据的灵活接口

    def __repr__(self) -> str:
        """
        返回类实例的字符串表示，不展开字段。

        :return: 类实例的简洁字符串表示，包含模块名、类名和内存地址。
        """
        return f"<{type(self).__module__}.{type(self).__qualname__} object at {hex(id(self))}>"


def build_lightning_datamodule(
    cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper
) -> pl.LightningDataModule:
    """
    Build the lightning datamodule from the config.
    :param cfg: Omegaconf dictionary.
    :param model: NN model used for training.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: Instantiated datamodule object.
    """
    # Build features and targets
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()

    # Build splitter
    splitter = build_splitter(cfg.splitter)

    # Create feature preprocessor
    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    # Create data augmentation
    augmentors = (
        build_agent_augmentor(cfg.data_augmentation)
        if "data_augmentation" in cfg
        else None
    )

    # Build dataset scenarios
    scenarios = build_scenarios(cfg, worker, model)

    # Create datamodule
    datamodule: pl.LightningDataModule = CustomDataModule(
        feature_preprocessor=feature_preprocessor,
        splitter=splitter,
        all_scenarios=scenarios,
        dataloader_params=cfg.data_loader.params,
        augmentors=augmentors,
        worker=worker,
        scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
        **cfg.data_loader.datamodule,
    )

    return datamodule


def build_lightning_module(
    cfg: DictConfig, torch_module_wrapper: TorchModuleWrapper
) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    # Create the complete Module
    if "custom_trainer" in cfg:
        model = instantiate(    # 用 hydra.utils.instantiate 或类似的工具函数来创建对象
            cfg.custom_trainer,
            model=torch_module_wrapper, # 传入封装好的 PyTorch 模型
            lr=cfg.lr,  # 学习率
            weight_decay=cfg.weight_decay,  # 权重衰减
            epochs=cfg.epochs,  # 训练轮数
            warmup_epochs=cfg.warmup_epochs,    # 预热轮数
        )
    else:
        objectives = build_objectives(cfg)
        metrics = build_training_metrics(cfg)
        model = LightningModuleWrapper(
            model=torch_module_wrapper,
            objectives=objectives,
            metrics=metrics,
            batch_size=cfg.data_loader.params.batch_size,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler if "lr_scheduler" in cfg else None,
            warm_up_lr_scheduler=cfg.warm_up_lr_scheduler
            if "warm_up_lr_scheduler" in cfg
            else None,
            objective_aggregate_mode=cfg.objective_aggregate_mode,
        )

    return cast(pl.LightningModule, model)


def build_custom_trainer(cfg: DictConfig) -> pl.Trainer:
    """
    根据配置callbacks和training_logger以构建 pl.Trainer 训练器。
    
    :param cfg: 包含配置信息的 omegaconf 字典
    :return: 构建的训练器对象
    """

    # 从配置中提取训练器参数
    params = cfg.lightning.trainer.params

    # 初始化回调函数列表，包括模型检查点、模型摘要、进度条和学习率监控
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints"),
            filename="{epoch}-{val_minFDE:.3f}",
            monitor=cfg.lightning.trainer.checkpoint.monitor,
            mode=cfg.lightning.trainer.checkpoint.mode,
            save_top_k=cfg.lightning.trainer.checkpoint.save_top_k,
            save_last=True,
        ),

        # PyTorch Lightning 提供的一个回调，它会在训练开始时打印出模型的结构和参数统计信息。max_depth=1 参数指定了只显示一层子模块的信息，这与你提供的输出格式相匹配
        # ┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
        # ┃   ┃ Name                         ┃ Type                 ┃ Params ┃
        # ┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
        # │ 0 │ model                        │ PlanningModel        │  4.1 M │
        # │ 1 │ model.pos_emb                │ FourierEmbedding     │  117 K │
        # │ 2 │ model.agent_encoder          │ AgentEncoder         │  672 K │
        # │ 3 │ model.map_encoder            │ MapEncoder           │  250 K │
        # │ 4 │ model.static_objects_encoder │ StaticObjectsEncoder │ 84.2 K │
        # │ 5 │ model.encoder_blocks         │ ModuleList           │  793 K │
        # │ 6 │ model.norm                   │ LayerNorm            │    256 │
        # │ 7 │ model.agent_predictor        │ AgentPredictor       │  223 K │
        # │ 8 │ model.planning_decoder       │ PlanningDecoder      │  1.9 M │
        # │ 9 │ collision_loss               │ ESDFCollisionLoss    │      0 │
        # └───┴──────────────────────────────┴──────────────────────┴────────┘
        RichModelSummary(max_depth=2),  
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # 根据配置选择合适的日志记录器
    if cfg.wandb.mode == "disable":
        # 如果禁用 wandb，则使用 TensorBoardLogger 进行日志记录
        training_logger = TensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version="",
            prefix="",
        )
    else:
        # 如果启用了 wandb，并且有指定的 artifact，则下载并设置 checkpoint 和 run_id
        if cfg.wandb.artifact is not None:
            os.system(f"wandb artifact get {cfg.wandb.artifact}")
            _, _, artifact = cfg.wandb.artifact.split("/")
            checkpoint = os.path.join(os.getcwd(), f"artifacts/{artifact}/model.ckpt")
            run_id = artifact.split(":")[0][-8:]
            cfg.checkpoint = checkpoint
            cfg.wandb.run_id = run_id

        # 使用 WandbLogger 进行日志记录，并根据配置决定是否恢复训练
        training_logger = WandbLogger(
            save_dir=cfg.group,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            log_model=cfg.wandb.log_model,
            resume=cfg.checkpoint is not None,
            id=cfg.wandb.run_id,
        )

    # 创建并返回训练器实例
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=training_logger,
        **params,
    )

    return trainer


def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info("Building training engine...")  # 记录构建训练引擎的日志信息

    trainer = build_custom_trainer(cfg) # 根据配置文件cfg创建PyTorch Lightning的Trainer对象

    # Create model. nuplan库函数
    torch_module_wrapper = build_torch_module_wrapper(cfg.model)    # cfg.model在train_pluto.yaml里配置为pluto_model

    # Build the datamodule. 使用build_lightning_datamodule函数创建数据模块datamodule
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)

    # Build lightning module. 使用build_lightning_module函数创建模型模块model
    model = build_lightning_module(cfg, torch_module_wrapper)

    # 将上述构建的Trainer、DataModule和Model封装成一个TrainingEngine对象并返回
    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine