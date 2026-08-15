#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
RewardProcess — LunarLander 奖励函数处理器。

================================================================================
奖励设计是强化学习最核心的部分。当前设计遵循 Gymnasium LunarLander-v3 的
shaping 奖励范式：

    final_reward = shaping_delta + fuel_cost     (非终止帧)
    final_reward = terminal_reward                (终止帧，覆盖上面)

其中：
    shaping = distance_penalty + velocity_penalty + angle_penalty + leg_contact_bonus
    shaping_delta = shaping(t) - shaping(t-1)     ← 势能差，驱动智能体向目标靠近
    fuel_cost = main_engine_cost + side_engine_cost

================================================================================
二次开发 — 奖励函数改进方向
================================================================================

当前奖励存在的问题（对 real 环境而言）：
    1. terminal_reward 只有 ±100 两档，太粗糙
       → 可以改为基于着陆质量的连续奖励
    2. 燃料消耗权重很低（0.30 / 0.03），智能体不关心省油
       → real 评分中燃料占 20%，应增大燃料惩罚
    3. shaping_delta 是一个稠密的每帧信号，可能让智能体只关注短期势能差
       而忽略长期规划
       → 可以尝试 sparse reward（只在 terminal 给奖励）或混合方案
    4. 没有对 landing precision 的直接奖励
       → 可以在 terminal 时根据最终 x 位置给予阶梯奖励

改进方案建议（按难度排序）：

    A. 调权重（简单）：
       - 增大 REWARD_MAIN_ENGINE_COST（如 0.3 → 1.0）
       - 增大 REWARD_SIDE_ENGINE_COST（如 0.03 → 0.1）
       - 增大 REWARD_LANDING（如 100 → 200）
       - 减小 REWARD_CRASH 绝对值（如 -100 → -200）加大坠毁惩罚

    B. 精细化终端奖励（中等）：
       - 成功着陆时，根据 |x| 位置给予分级奖励：
         |x| < 0.1 → +200
         |x| < 0.25 → +150
         |x| < 0.5 → +100
       - 根据剩余燃料给予额外加分

    C. 引入 real_score 作为奖励（已实现）：
       - real 环境的 final_score（fuel/precision/stability 加权分）
         已在 RealLunarLanderWrapper.step() 中混入 episode 结束帧奖励
       - 见 agent_ppo/feature/real_env.py

    D. 课程学习（已实现）：
       - 先在 base 环境训练到收敛，再用 train_real.py --init-model 加载该模型
         在 real 环境 fine-tune（见 train_workflow.py 的 init_model_path）

================================================================================
"""

from __future__ import annotations

import numpy as np

from agent_ppo.conf.conf import Config

try:
    import gymnasium
except ImportError as exc:
    raise RuntimeError("This project requires gymnasium. Install it with: pip install gymnasium[box2d]") from exc

# numpy 2.x 兼容：np.bool8 在 2.x 中已废弃
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


# =============================================================================
# RewardProcess — 奖励计算核心
# =============================================================================
class RewardProcess:
    """
    显式的 LunarLander 奖励处理器。

    将奖励拆分为独立的小方法（_reward_xxx），方便逐项调参和调试。
    每帧的最终奖励通过 compute() 计算，返回标量和各分项字典。
    """

    def __init__(self):
        # prev_shaping: 上一帧的 shaping 值，用于计算 shaping_delta
        # shaping_delta > 0 表示智能体正在靠近目标（势能降低）
        self.prev_shaping = None

    def reset(self, obs=None, env=None):
        """
        每局开始时重置 shaping 基准。
        优先从环境的 unwrapped.prev_shaping 读取（Gymnasium 内部维护），
        否则根据初始观测计算。
        """
        if env is not None and hasattr(env.unwrapped, "prev_shaping"):
            self.prev_shaping = float(env.unwrapped.prev_shaping)
        elif obs is not None:
            self.prev_shaping = self._calculate_shaping(obs)
        else:
            self.prev_shaping = None

    def compute(self, obs, action, env_reward, terminated=False):
        """
        计算单帧奖励。

        参数:
            obs: 8 维观测向量
            action: 执行的离散动作 (0~3)
            env_reward: 环境原始奖励（仅用于判断 ±100 terminal 事件）
            terminated: 是否因着陆/坠毁而终止

        说明:
            real 环境的 final_score 混入奖励不在这里处理，而是在
            RealLunarLanderWrapper.step()（见 agent_ppo/feature/real_env.py），
            因为该评分要等环境步进完成后才能算出来。

        返回:
            (final_reward, components_dict)
        """
        state = np.asarray(obs, dtype=np.float32)

        # ---- 计算当前帧的 shaping 和 shaping_delta ----
        shaping = self._calculate_shaping(state)
        shaping_delta = 0.0 if self.prev_shaping is None else float(shaping - self.prev_shaping)
        self.prev_shaping = shaping

        # ---- 计算所有奖励分项 ----
        components = {
            "distance": self._reward_distance(state),           # 距离惩罚
            "velocity": self._reward_velocity(state),           # 速度惩罚
            "angle": self._reward_angle(state),                 # 角度惩罚
            "left_leg_contact": self._reward_left_leg_contact(state),   # 左腿着地奖励
            "right_leg_contact": self._reward_right_leg_contact(state), # 右腿着地奖励
            "shaping": float(shaping),                          # 总 shaping
            "shaping_delta": float(shaping_delta),              # shaping 差分
            "main_engine_cost": self._reward_main_engine_cost(action),   # 主发动机燃料成本
            "side_engine_cost": self._reward_side_engine_cost(action),   # 侧发动机燃料成本
            "terminal_reward": self._reward_terminal(env_reward, terminated, state),  # 终止奖励
            "env_reward": float(env_reward),                    # 环境原始奖励（仅用于调试）
        }

        # ---- 组合最终奖励 ----
        # 非终止帧：奖励 = shaping_delta + 燃料成本
        #   shaping_delta > 0 → 正在靠近目标 → 正奖励
        #   燃料成本 < 0 → 开引擎有代价 → 负奖励
        # 终止帧：奖励 = terminal_reward（覆盖上面所有）
        #   +100 → 成功着陆，-100 → 坠毁
        total = (
            components["shaping_delta"]
            + components["main_engine_cost"]
            + components["side_engine_cost"]
        )
        if components["terminal_reward"] != 0.0:
            total = components["terminal_reward"]
        components["total"] = float(total)
        final_reward = Config.REWARD_SCALE * total + Config.REWARD_BIAS
        return float(final_reward), components

    # ------------------------------------------------------------------
    # 各奖励分量（以 _reward_ 开头，方便 A/B 测试时增减）
    # ------------------------------------------------------------------

    def _reward_distance(self, state):
        """距离惩罚 = -100 * sqrt(x² + y²)

        登陆器离目标原点越远，惩罚越大。
        这是 shaping 中权重最大的项，驱动智能体向原点移动。
        """
        return float(Config.REWARD_DISTANCE_WEIGHT * np.sqrt(state[0] * state[0] + state[1] * state[1]))

    def _reward_velocity(self, state):
        """速度惩罚 = -100 * sqrt(vx² + vy²)

        速度越快惩罚越大，鼓励缓慢、受控的下降。
        """
        return float(Config.REWARD_VELOCITY_WEIGHT * np.sqrt(state[2] * state[2] + state[3] * state[3]))

    def _reward_angle(self, state):
        """角度惩罚 = -100 * |angle|

        机体倾斜越大惩罚越大，鼓励保持竖直姿态着陆。
        """
        return float(Config.REWARD_ANGLE_WEIGHT * abs(state[4]))

    def _reward_left_leg_contact(self, state):
        """左腿接触奖励 = 10 * left_contact

        left_contact ∈ [0, 1]，1 表示左腿触地。
        只有触地才能触发 landing 判定。
        """
        return float(Config.REWARD_LEG_CONTACT_BONUS * state[6])

    def _reward_right_leg_contact(self, state):
        """右腿接触奖励 = 10 * right_contact

        同左腿逻辑。
        """
        return float(Config.REWARD_LEG_CONTACT_BONUS * state[7])

    def _reward_main_engine_cost(self, action):
        """主发动机燃料成本

        每次开主发动机扣 REWARD_MAIN_ENGINE_COST（当前 0.90）。
        主发动机推力大、耗油多 → 惩罚更重。
        """
        return float(-self._main_engine_power(action) * Config.REWARD_MAIN_ENGINE_COST)

    def _reward_side_engine_cost(self, action):
        """侧发动机燃料成本

        每次开侧发动机扣 REWARD_SIDE_ENGINE_COST（当前 0.30）。
        侧发动机推力小、耗油少 → 惩罚更轻。
        """
        return float(-self._side_engine_power(action) * Config.REWARD_SIDE_ENGINE_COST)

    def _reward_terminal(self, env_reward, terminated=False, state=None):
        """
        终止奖励：只在 episode 结束时触发。

        着陆成功时，根据 |x| 位置给予分级奖励：
          基础分 = REWARD_LANDING (=100)
          精度加成 = REWARD_LANDING_PRECISION_BONUS × (1 - |x| / REWARD_LANDING_X_RANGE)
          总奖励 = 基础分 + 精度加成

          示例（BONUS=100, X_RANGE=0.3）：
            |x|=0    → 100 + 100 = 200     ← 完美正中靶心
            |x|=0.15 → 100 + 50  = 150     ← 偏了15cm
            |x|≥0.3  → 100 + 0   = 100     ← 偏出精度范围
          加上 real_score（最高 +30）：terminal 最大 = 230

        坠毁时：固定 REWARD_CRASH (= -100)
        """
        if not terminated:
            return 0.0
        if env_reward >= Config.REWARD_LANDING:
            # 着陆成功 — 根据 |x| 位置计算精度加成
            base = float(Config.REWARD_LANDING)
            if state is not None:
                abs_x = abs(float(np.asarray(state).reshape(-1)[0]))
                ratio = 1.0 - min(abs_x / max(Config.REWARD_LANDING_X_RANGE, 1e-6), 1.0)
                bonus = float(Config.REWARD_LANDING_PRECISION_BONUS) * ratio
                return base + bonus
            return base
        if env_reward <= Config.REWARD_CRASH:
            return float(Config.REWARD_CRASH)
        return 0.0

    def _calculate_shaping(self, obs):
        """
        计算 shaping = 距离惩罚 + 速度惩罚 + 角度惩罚 + 腿接触奖励。

        shaping 是一个"势能函数"：值越大（越接近 0）表示状态越好。
        shaping_delta = shaping(t) - shaping(t-1)：
          正值 → 状态在改善（靠近目标、减速、扶正）→ 正奖励
          负值 → 状态在恶化 → 负奖励
        """
        state = np.asarray(obs, dtype=np.float32)
        return float(
            self._reward_distance(state)
            + self._reward_velocity(state)
            + self._reward_angle(state)
            + self._reward_left_leg_contact(state)
            + self._reward_right_leg_contact(state)
        )

    @staticmethod
    def _main_engine_power(action):
        """主发动机是否开启：action == 2"""
        action = int(np.asarray(action).reshape(-1)[0])
        return 1.0 if action == 2 else 0.0

    @staticmethod
    def _side_engine_power(action):
        """侧发动机是否开启：action == 1 或 3"""
        action = int(np.asarray(action).reshape(-1)[0])
        return 1.0 if action in (1, 3) else 0.0


# =============================================================================
# LunarLanderRewardWrapper — Gymnasium 包装器
# =============================================================================
class LunarLanderRewardWrapper(gymnasium.Wrapper):
    """
    将 RewardProcess 嵌入 Gymnasium 环境。

    这个 Wrapper 在 env_factory 中作为最内层包装应用。
    每个 step() 调用：
        1. 执行原始环境的 step()
        2. 用 RewardProcess 重新计算奖励
        3. 返回新奖励 + 原始观测 + 分项信息（在 info 字典中）
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward_process = RewardProcess()

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.reward_process.reset(obs=obs, env=self)
        return reset_out

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        # 用自定义奖励覆盖环境原始奖励
        reward, components = self.reward_process.compute(
            obs, action, env_reward, terminated
        )
        info = dict(info)
        info["reward_components"] = components   # 保存分项，方便调试和分析
        return obs, reward, terminated, truncated, info
