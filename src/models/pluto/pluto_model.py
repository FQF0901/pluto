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
        - data: 包含agent, map, reference_line, static_objects, current_state, origin, angle, cost_map的字典。

        返回:
        - out: 包含模型输出的字典，包括轨迹、概率和预测等信息。
        """
        # 1.1 为了PE提取agent的当前位、方向和掩码信息。agent的[位置坐标、航向角、速度向量、感知边界框的尺寸、该帧的观察状态]
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1] # [batch_size, N_agents, T_future, x/y]: [4, 49, 101, 2] -> [4, 49, 2]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]  # [4, 49, 101] -> [4, 49]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]    # self.history_steps=21, [4, 49, 101] -> [4, 49, 21]
        
        # 提取地图多边形中心和掩码信息
        polygon_center = data["map"]["polygon_center"]  # [batch_size, N_polygons, x/y/yaw], [4, 149, 3]
        polygon_mask = data["map"]["valid_mask"]    # [batch_size, N_polygons, N_points], [4, 149, 20]

        # 获取批次大小和代理数量
        bs, A = agent_pos.shape[0:2]    # 4, 49

        # 合并代理位置和多边形中心，合并代理方向和多边形方向，并进行角度归一化
        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)   # [batch_size, N_agents+N_polygons, x/y] -> [4, 198, 2]
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)   # [batch_size, N_agents+N_polygons, yaw] -> [4, 198]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi # [4, 198]
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)    # [4, 198, 3]

        # 计算代理和多边形的有效掩码
        agent_key_padding = ~(agent_mask.any(-1))   # [batch_size, N_agents], [4, 49]
        polygon_key_padding = ~(polygon_mask.any(-1))   # [batch_size, N_polygons], [4, 149]
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)  # [batch_size, N_agents+N_polygons], [4, 198]

        # 1.2 使用编码器分别对代理、地图多边形和静态物体进行编码
        x_agent = self.agent_encoder(data)  # FPN: [4, 49, 128]: [batch_size, n_agents, dim]
        x_polygon = self.map_encoder(data)  # PointNet: [4, 149, 128]
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)    # MLP: [4, 17, 128], [4, 17, 3], [4, 17]。static obj的[位置坐标、航向角、感知边界框的尺寸]

        # 合并所有编码后的特征，为啥没有自车？
        x = torch.cat([x_agent, x_polygon, x_static], dim=1)    # [4, 215, 128]

        # 合并位置信息和静态物体位置信息，并生成PE
        pos = torch.cat([pos, static_pos], dim=1)   # [4, 215, 3]
        pos_embed = self.pos_emb(pos)   # [4, 215, 128]

        # 1.3 合并所有掩码信息
        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)    # [4, 215]
        x = x + pos_embed   # [4, 215, 128] + [4, 215, 128] = [4, 215, 128]

        # 2. 通过编码器块进行进一步处理
        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)    # [4, 215, 128]: [batch_size, n_agent+n_static_obj+n_polygon, dim_feature]

        # 3. 使用代理预测器对代理的未来位置进行预测
        prediction = self.agent_predictor(x[:, 1:A])    # A=49, prediction:[4, 48, 80, 6]貌似[batch_size, n_agent-1, T_future, 6_infos]

        # 4. 检查是否有可用的参考线
        ref_line_available = data["reference_line"]["position"].shape[1] > 0    # true

        # 如果有可用的参考线，则使用规划解码器生成轨迹和概率
        if ref_line_available:
            trajectory, probability = self.planning_decoder(data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}) # [4, 3, 12, 80, 6], [4, 3, 12]
        else:
            trajectory, probability = None, None

        # 5. 构建输出字典，这些全都是相对轨迹
        out = {
            "trajectory": trajectory,   # [4, 3, 12, 80, 6]
            "probability": probability,  # (bs, R, M): [4, 3, 12]
            "prediction": prediction,  # (bs, A-1, T, 2): [4, 48, 80, 6]
        }

        # 如果使用隐藏层投影，则添加到输出字典中。貌似用在对比学习中
        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        # 如果使用无参考线的轨迹，则生成并添加到输出字典中
        if self.ref_free_traj:
            # x[:, 0] 提取了每个批次中第一个位置的特征（通常是自车或场景中的关键代理）的特征表示
            # self.ref_free_decoder：这是一个多层感知机（MLP）模型，用于生成不依赖于参考线的未来轨迹预测
            # reshape 操作，将 MLP 输出的扁平化张量重新组织成形状为 [batch_size, future_steps, 4] 的张量
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(bs, self.future_steps, 4)
            out["ref_free_trajectory"] = ref_free_traj

        # 6. 如果不在训练模式下，则生成最终输出轨迹、预测和概率
        if not self.training:   # train的时候为false
            # 6.1 ref_traj
            if self.ref_free_traj:  # train的时候为false
                ref_free_traj_angle = torch.arctan2(ref_free_traj[..., 3], ref_free_traj[..., 2])   # 使用 torch.arctan2 计算参考自由轨迹角度
                ref_free_traj = torch.cat([ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1)  # 将原始轨迹中的位置信息与计算得到的角度信息拼接在一起
                out["output_ref_free_trajectory"] = ref_free_traj

            # 6.2 绝对坐标的agent traj
            output_prediction = torch.cat(
                [
                    # prediction[..., :2]：这是模型预测的未来轨迹中每个时间步的位置偏移量（x 和 y 坐标）
                    # agent_pos[:, 1:A, None]：这是当前代理（agent）的位置坐标。None 的作用是扩展维度以匹配 prediction 的形状
                    # 将预测的未来位置偏移量与当前代理的位置相加，得到绝对位置坐标
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    # torch.atan2(prediction[..., 3], prediction[..., 2])：使用 arctan2 函数计算预测轨迹中每个时间步的方向角度。unsqueeze(-1)：扩展维度以匹配其他张量的形状
                    # agent_heading[:, 1:A, None, None]：这是当前代理的航向角。同样通过 None 扩展维度以匹配形状
                    # 将预测的方向角度与当前代理的航向角相加，得到绝对方向角度
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1) + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],   # 这部分直接取自 prediction 的剩余部分，通常包含其他特征（例如速度、加速度等），保持不变
                ],
                dim=-1,)   # [4, 48, 80, 5]
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                # 6.3 处理参考线的有效性掩码，并对无效的参考线对应的概率进行填充，以确保在后续选择最佳轨迹时不会选择这些无效的参考线
                # data["reference_line"]["valid_mask"]：这是一个布尔张量，形状为 [batch_size, num_reference_lines, future_steps]
                # .any(-1)：对最后一个维度（即 future_steps）进行逻辑或操作。只要该维度上有一个元素为 True，结果就为 True， 生成一个形状为 [batch_size, num_reference_lines] 的张量
                # ~：取反操作，最终得到的 r_padding_mask 是一个布尔张量，形状为 [batch_size, num_reference_lines]
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)  # [4, 3]，infer时[1, 1]
                # r_padding_mask.unsqueeze(-1)：扩展 r_padding_mask 的维度，使其形状变为 [batch_size, num_reference_lines, 1]，以便与 probability 张量的形状匹配
                # probability.masked_fill_：这是一个原地操作，用于将 probability 中对应于无效参考线的位置填充为一个非常小的值（如 -1e6）。这样做的目的是确保在后续选择最佳轨迹时，这些无效的参考线不会被选中
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)    # infer时[1, 1, 12]

                # trajectory[..., 3] 和 trajectory[..., 2]：分别表示轨迹中每个时间步的速度或方向分量（通常是 x 和 y 方向的速度分量）
                # torch.atan2(y, x)：这个函数返回的是从正 x 轴到点 (x, y) 的向量之间的角度，结果范围在 ([-π, π]) 之间。通过 atan2 计算出这些点的方向角度
                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2]) # [4, 3, 12, 80], infer时[1, 1, 12, 80]
                # 沿最后一个维度（即特征维度）拼接位置坐标和角度信息
                out_trajectory = torch.cat([trajectory[..., :2], angle.unsqueeze(-1)], dim=-1)   # [4, 3, 12, 80, 3], infer时[1, 1, 12, 80, 3]

                bs, R, M, T, _ = out_trajectory.shape   # 4, 3, 12, 80, infer时[1, 1, 12, 80, 3]
                flattened_probability = probability.reshape(bs, R * M)  # [4, 36], infer时[1, 12]
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[torch.arange(bs), flattened_probability.argmax(-1)]   # [4, 80, 3], infer时[1, 80, 3]

                out["output_trajectory"] = best_trajectory  # [4, 80, 3], infer时[1, 80, 3]
                out["candidate_trajectories"] = out_trajectory  # [4, 3, 12, 80, 3], infer时[1, 1, 12, 80, 3]
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(1, 0, 0, self.future_steps, 3)

        return out
