"""
环境工厂 — 创建 LunarLander 环境的统一入口。

================================================================================
双环境模式

    BASE_ENV_MODE ("base")：原始 Gymnasium LunarLander-v3
        └── 环境结构: gym.make → LunarLanderRewardWrapper

    REAL_ENV_MODE ("real")：仿真真实环境
        └── 环境结构: gym.make → LunarLanderRewardWrapper → RealLunarLanderWrapper
            （从内到外的包装顺序）

包装顺序很重要：
    - RewardWrapper 必须最内层（直接包装 gym 环境），因为它替换了环境奖励
    - RealLunarLanderWrapper 在最外层，对策略提供噪声观测、执行延迟动作
    - 策略所见到的观测是 RealLunarLanderWrapper.step() 返回的 noisy_obs

================================================================================
"""

from __future__ import annotations

import gymnasium as gym

from agent_ppo.conf.conf import Config
from agent_ppo.feature.real_env import RealLunarLanderWrapper
from agent_ppo.feature.reward_process import LunarLanderRewardWrapper


# 环境模式常量
BASE_ENV_MODE = "base"
REAL_ENV_MODE = "real"
ENV_MODES = (BASE_ENV_MODE, REAL_ENV_MODE)


def run_name(env_mode: str, env_id: str | None = None) -> str:
    """
    生成运行名称。

    base → "LunarLander-v3"
    real → "LunarLander-v3-real"

    这个名称用于日志目录命名和模型文件命名。
    """
    env_id = env_id or Config.ENV_ID
    return env_id if env_mode == BASE_ENV_MODE else f"{env_id}-{env_mode}"


def make_lunarlander_env(env_mode: str = BASE_ENV_MODE, render_mode: str | None = None):
    """
    创建单个 LunarLander 环境实例（带完整的包装器链）。

    参数:
        env_mode: "base" 或 "real"
        render_mode: None（训练）/ "rgb_array"（回放）/ "human"（直接显示）

    返回:
        包装好的 Gymnasium 环境

    环境链（以 real 为例）：
        RealLunarLanderWrapper            ← 最外层：噪声观测 + 延迟动作 + 阵风
          └── LunarLanderRewardWrapper    ← 中层：自定义奖励计算
                └── gym.make("LunarLander-v3")  ← 内层：原始 Box2D 环境

    注意：
        此函数返回单个环境，训练时通过 make_vec_env() 复制为 VecEnv。
        make_vec_env 会多次调用此函数（每进程一次）。
    """
    if env_mode not in ENV_MODES:
        raise ValueError(f"Unsupported env_mode: {env_mode}. Expected one of {ENV_MODES}")

    # 1. 创建原始 LunarLander 环境
    env = gym.make(Config.ENV_ID, render_mode=render_mode)

    # 2. 套上奖励包装器（必须在内层）
    env = LunarLanderRewardWrapper(env)

    # 3. 如果是 real 模式，套上真实环境包装器
    if env_mode == REAL_ENV_MODE:
        env = RealLunarLanderWrapper(env)

    return env
