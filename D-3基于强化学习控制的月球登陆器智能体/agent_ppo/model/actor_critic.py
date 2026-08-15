#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Actor-Critic 网络结构 — LunarLander PPO 的神经网络实现。

================================================================================
当前架构：双 MLP（独立 Actor 和 Critic 网络）

    Actor:  8 → [64, 64] → 4 (action logits)
    Critic: 8 → [64, 64] → 1 (state value)

这是 Stable-Baselines3 MlpPolicy 的标准结构：
    - Actor 和 Critic 各自拥有独立的隐藏层（不共享参数）
    - 都用 Tanh 激活函数 + 正交初始化
    - Actor 输出头用很小 gain（0.01）防止初始策略过于极端
    - Critic 输出头用标准 gain（1.0）

================================================================================
二次开发 — 网络结构改进方向
================================================================================

    A. 增大网络容量（应对 real 环境）：
       real 环境有噪声和延迟，8→[64,64]→4 的简单 MLP 可能不够。
       可以尝试：
           [128, 128]     — 更宽，更多神经元
           [256, 128, 64] — 更深，多层抽象
           [128, 128, 64] — 宽深折中

    B. 共享特征提取层：
       让 Actor 和 Critic 共享前面的隐藏层，减少参数、加速训练。
       SB3 支持 net_arch=dict(pi=[64,64], vf=[64,64]) 的独立结构，
       也支持 net_arch=[dict(vf=[64], pi=[64])] 的共享结构。
       共享结构参数更少，对 small data 可能更好。

    C. 尝试其他激活函数：
       Tanh 在深层网络中容易梯度饱和。
       ReLU/ELU 计算更快、梯度更稳定。
       调整方式：Config.ACTIVATION_FN = "nn.ReLU" 或 "nn.ELU"

    D. 引入 LSTM/GRU（处理延迟）：
       real 环境有 8 帧延迟，带记忆的循环网络可能比 MLP 更有优势。
       这需要将 MlpPolicy 换成 SB3 的 recurrent policy。

    E. 集成学习：
       训练多个不同随机种子的模型，在推理时用多数投票选择动作。
       可提高 real 环境下的鲁棒性。
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


# =============================================================================
# 工具函数
# =============================================================================

def resolve_nn_activation(activation: str):
    """
    将配置中的激活函数字符串解析为 PyTorch 激活类。

    支持: relu / tanh / elu / selu / lrelu / sigmoid
    前缀 "nn." 会被自动去掉（兼容 SB3 的配置格式）。
    """
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
    }
    key = activation.lower().replace("nn.", "")
    if key not in activation_map:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_map.keys())}")
    return activation_map[key]


def _make_fc(in_dim, out_dim, gain=1.41421):
    """
    创建一个全连接层，使用正交初始化和零偏置。

    gain 参数控制权重的初始化幅度：
        gain=1.41421（sqrt(2)）：ReLU 推荐值
        gain=1.0：Tanh/Sigmoid 推荐值
        gain=0.01：Actor 输出头（保证初始策略接近均匀分布）
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


# =============================================================================
# ActorCritic — 双 MLP Actor-Critic
# =============================================================================

class ActorCritic(nn.Module):
    """
    显式 MLP Actor-Critic 结构，与 SB3 MlpPolicy 等价。

    属性：
        is_recurrent = False  — SB3 用此标志判断是否为循环网络
        actor_backbone        — Actor 的特征提取 MLP
        critic_backbone       — Critic 的特征提取 MLP
        actor_head            — Actor 的输出层（logits）
        critic_head           — Critic 的输出层（scalar value）
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int = Config.OBS_DIM,              # 观测维度 = 8
        num_critic_obs: int = Config.OBS_DIM,        # Critic 输入维度（通常与 Actor 相同）
        num_actions: int = Config.ACTION_NUM,        # 动作数量 = 4
        actor_hidden_dims=None,                      # Actor 隐藏层维度列表
        critic_hidden_dims=None,                     # Critic 隐藏层维度列表
        activation: str = Config.ACTIVATION_FN,      # 激活函数
        **kwargs,
    ):
        super().__init__()
        self.model_name = "lunarlander_ppo"

        actor_hidden_dims = list(actor_hidden_dims or Config.ACTOR_HIDDEN_LAYERS)
        critic_hidden_dims = list(critic_hidden_dims or Config.CRITIC_HIDDEN_LAYERS)
        activation_fn = resolve_nn_activation(activation)

        # 构建 Actor 网络：backbone + head
        self.actor_backbone = self._build_mlp(num_obs, actor_hidden_dims, activation_fn)
        self.actor_head = _make_fc(actor_hidden_dims[-1], num_actions, gain=0.01)
        # gain=0.01 → 初始 logits 接近 0 → softmax 后接近均匀分布
        # 这确保训练初期智能体随机探索，不会偏向某个动作

        # 构建 Critic 网络：backbone + head
        self.critic_backbone = self._build_mlp(num_critic_obs, critic_hidden_dims, activation_fn)
        self.critic_head = _make_fc(critic_hidden_dims[-1], Config.VALUE_NUM, gain=1.0)
        # gain=1.0 → 标准初始化，Value 网络不需要特殊处理

    def _build_mlp(self, input_dim, hidden_dims, activation_fn):
        """
        构建 MLP 骨干网络。

        结构：Linear → Act → Linear → Act → ...（交替排列）
        不包含输出层（head），只做特征提取。
        """
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(_make_fc(last_dim, hidden_dim))
            layers.append(activation_fn())
            last_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, obs, inference=False):
        """
        前向传播。

        参数:
            obs: 观测张量，shape (batch, 8)
            inference: 是否推理模式（当前未使用，SB3 兼容保留）

        返回:
            [logits, value]
            logits: shape (batch, 4)，每个动作的未归一化分数
            value: shape (batch, 1)，状态价值估计
        """
        obs = obs.to(torch.float32)
        logits = self.actor_head(self.actor_backbone(obs))
        value = self.critic_head(self.critic_backbone(obs))
        return [logits, value]

    def set_train_mode(self):
        """进入训练模式（启用 dropout、batch_norm 等）。"""
        self.train()

    def set_eval_mode(self):
        """进入评估模式。"""
        self.eval()


# =============================================================================
# LunarLanderPPOModelSpec — 模型规格描述
# =============================================================================

@dataclass(frozen=True)
class LunarLanderPPOModelSpec:
    """
    模型架构元数据，方便在 agent.describe() 中查看网络信息。

    frozen=True → 创建后不可修改，防止意外改动。
    """

    policy: str = Config.POLICY
    observation_dim: int = Config.OBS_DIM
    action_num: int = Config.ACTION_NUM
    value_num: int = Config.VALUE_NUM
    actor_hidden_layers: tuple[int, int] = tuple(Config.ACTOR_HIDDEN_LAYERS)
    critic_hidden_layers: tuple[int, int] = tuple(Config.CRITIC_HIDDEN_LAYERS)
    activation: str = Config.ACTIVATION_FN

    def describe(self) -> str:
        """返回人类可读的模型信息字符串。"""
        return (
            f"Policy: {self.policy}\n"
            f"Input: {self.observation_dim}D LunarLander state\n"
            f"Actor MLP: {self.actor_hidden_layers} -> {self.action_num} action logits\n"
            f"Critic MLP: {self.critic_hidden_layers} -> {self.value_num} state value\n"
            f"Activation: {self.activation}\n"
        )


# 别名，供外部 import
Model = ActorCritic
