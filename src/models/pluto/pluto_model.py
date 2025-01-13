from copy import deepcopy
import math

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .layers.transformer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder import PlanningDecoder
from .layers.mlp_layer import MLPLayer

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:

        """
        初始化模型参数和组件。

        :param dim: 模型的维度。
        :param state_channel: 状态通道的数量。
        :param polygon_channel: 多边形通道的数量。
        :param history_channel: 历史通道的数量。
        :param history_steps: 历史步数。
        :param future_steps: 未来步数。
        :param encoder_depth: 编码器的深度。
        :param decoder_depth: 解码器的深度。
        :param drop_path: 路径丢弃的概率。
        :param dropout: Dropout的概率。
        :param num_heads: 注意力头的数量。
        :param num_modes: 模式数量。
        :param use_ego_history: 是否使用自我历史。
        :param state_attn_encoder: 是否使用状态注意力编码器。
        :param state_dropout: 状态Dropout的概率。
        :param use_hidden_proj: 是否使用隐藏投影。
        :param cat_x: 是否连接输入x。
        :param ref_free_traj: 是否使用参考自由轨迹。
        :param feature_builder: 特征构建器实例。
        """

        # 调用父类的初始化方法进行基础设置
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        # 模型参数初始化
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj

        # 位置编码使用傅里叶嵌入
        self.pos_emb = FourierEmbedding(3, dim, 64)

        # 构建agent编码器，处理历史信息和状态信息
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        # 构建map编码器，处理地图信息
        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        # 构建static objects编码器，处理静态物体信息
        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        # 构建编码器模块列表，使用Transformer编码层
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        # 使用层归一化
        self.norm = nn.LayerNorm(dim)

        # 构建agent预测器，预测未来轨迹
        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        # 构建planning解码器，进行多模式规划
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        # 如果使用隐藏投影，则构建隐藏投影层
        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        # 如果使用参考自由轨迹，则构建参考自由解码器
        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        # 对整个模型的应用初始化权重方法
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化模型参数。
        
        该方法根据输入模块的不同类型，使用不同的初始化策略。特别是，对于线性层、层归一化、批量归一化和嵌入层，
        它们分别采用了不同的权重和偏置初始化方法。
        
        参数:
        - m: 模型模块。这是一个PyTorch模块实例，其权重和偏置需要被初始化。
        """
        if isinstance(m, nn.Linear):    # 对于线性层（全连接层），使用Xavier均匀初始化方法初始化权重。
            torch.nn.init.xavier_uniform_(m.weight)
            # 如果线性层的偏置项存在，则将偏置项初始化为0。
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):   # 对于层归一化，将偏置项初始化为0，权重初始化为1，以保持输入的均值和方差不变。
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d): # 对于一维批量归一化，将权重初始化为1，偏置项初始化为0，以维持输入的均值和方差不变。
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):   # 对于嵌入层，使用均值为0，标准差为0.02的正态分布随机初始化权重。
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, data):
        """
        处理输入数据，生成模型预测输出。

        参数:
        - data: 包含代理位置、方向、有效掩码，地图多边形中心和有效掩码等信息的字典。

        返回:
        - out: 包含模型输出的字典，包括轨迹、概率和预测等信息。
        """
        # 提取代理的历史位置、方向和掩码信息
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        
        # 提取地图多边形中心和掩码信息
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        # 获取批次大小和代理数量
        bs, A = agent_pos.shape[0:2]

        # 合并代理位置和多边形中心，合并代理方向和多边形方向，并进行角度归一化
        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        # 计算代理和多边形的有效掩码
        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        # 使用编码器分别对代理、地图多边形和静态物体进行编码
        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        # 合并所有编码后的特征
        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        # 合并位置信息和静态物体位置信息，并生成位置嵌入
        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        # 合并所有掩码信息
        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        # 通过编码器块进行进一步处理
        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        # 使用代理预测器对代理的未来位置进行预测
        prediction = self.agent_predictor(x[:, 1:A])

        # 检查是否有可用的参考线
        ref_line_available = data["reference_line"]["position"].shape[1] > 0

        # 如果有可用的参考线，则使用规划解码器生成轨迹和概率
        if ref_line_available:
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )
        else:
            trajectory, probability = None, None

        # 构建输出字典
        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
        }

        # 如果使用隐藏层投影，则添加到输出字典中
        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        # 如果使用无参考线的轨迹，则生成并添加到输出字典中
        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        # 如果不在训练模式下，则生成最终输出轨迹、预测和概率
        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )

        return out
