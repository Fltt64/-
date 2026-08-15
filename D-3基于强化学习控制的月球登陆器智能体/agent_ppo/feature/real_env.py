#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
RealEnv — 仿真真实世界的 LunarLander 环境包装器。

================================================================================
在标准 Gymnasium LunarLander-v3 之上叠加三种真实世界扰动：

    1. 传感器高斯噪声 — 模拟 GPS/IMU 读数不精确
       每帧在 8 维观测上叠加 σ=0.10 的高斯噪声
       噪声后 clip 到观测空间范围内

    2. 控制延迟 — 模拟推进器点火到产生推力的延迟
       action 送入一个固定长度的 FIFO 队列
       实际执行的是 N 帧之前排入的动作
       未填充时执行默认动作（不喷火）

    3. 随机阵风 — 模拟着陆过程中的突风扰动
       每帧以 18% 概率触发一次阵风检查
       阵风持续 8~32 帧，水平和垂直方向独立采样
       通过 Box2D 的 ApplyForceToCenter 直接施加力

================================================================================
二次开发 — real 环境改进方向
================================================================================

    A. 噪声模型改进：
       - 当前是固定 σ 的高斯噪声，可改为随高度变化（低空噪声小、高空噪声大）
       - 可加入传感器偏置（bias），模拟 IMU 零漂
       - 可对特定维度加不同噪声（如角度噪声更大，因为陀螺仪在振动下更不准）

    B. 延迟模型改进：
       - 当前是固定 8 帧延迟，可改为随机延迟（6~12 帧）
       - 可对左/右/主发动机设置不同延迟（如主发动机延迟更长）

    C. 阵风模型改进：
       - 当前是全局均匀触发，可改为与高度相关的阵风剖面
       - 可加入持续侧风（bias）而不仅仅是随机阵风
       - 可模拟地面效应（近地时有额外湍流）

    D. 执行器故障：
       - 可加入随机发动机失效（一定概率推进器不点火）
       - 可加入推力衰减（随燃料消耗推力下降）

================================================================================
"""

from __future__ import annotations

from collections import deque

import gymnasium
import numpy as np

from agent_ppo.conf.conf import Config


# =============================================================================
# RealLunarLanderWrapper — 真实环境包装器
# =============================================================================
class RealLunarLanderWrapper(gymnasium.Wrapper):
    """
    仿真真实世界的 LunarLander 包装器。

    包装顺序（从外到内）：
        RealLunarLanderWrapper  ← 噪声、延迟、阵风（本类）
            LunarLanderRewardWrapper  ← 自定义奖励
                gym.make("LunarLander-v3")  ← 原始环境

    包装后的策略接收的是含噪声的观测，执行的动作经过延迟管线。
    环境原始奖励不变，最终评分通过 info["real_score"] 报告。
    """

    def __init__(
        self,
        env,
        obs_noise_std: float = Config.REAL_OBS_NOISE_STD,
        action_delay_steps: int = Config.REAL_ACTION_DELAY_STEPS,
    ):
        super().__init__(env)
        self.obs_noise_std = float(obs_noise_std)
        self.action_delay_steps = max(0, int(action_delay_steps))

        # ---- 控制延迟 FIFO 队列 ----
        # 长度 = delay_steps + 1：1 个当前动作 + delay_steps 个待执行动作
        # 新动作进队尾，实际执行从队头取
        self.action_queue = deque(maxlen=self.action_delay_steps + 1)

        # ---- 独立随机数生成器 ----
        # 使用独立 seed（seed + offset），确保噪声和训练环境的随机性不耦合
        self.rng = np.random.default_rng()

        # ---- 评分追踪器 ----
        self.score_tracker = RealScoreTracker()

        # ---- 阵风状态 ----
        self.gust_steps_left = 0                # 剩余阵风帧数
        self.gust_force = np.zeros(2, dtype=np.float32)  # 当前阵风力 [Fx, Fy]

    # ------------------------------------------------------------------
    # Gymnasium Wrapper 接口
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        """
        重置环境，同时重置所有 real 状态。

        流程：
            1. 用独立种子初始化噪声 RNG
            2. 清空动作延迟队列，填充默认动作
            3. 重置评分追踪器和阵风状态
            4. 对初始观测加噪声后返回
        """
        seed = kwargs.get("seed")
        if seed is not None:
            # 噪声使用独立的随机种子 = 环境种子 + 固定偏移量
            self.rng = np.random.default_rng(seed + Config.REAL_NOISE_SEED_OFFSET)

        reset_out = self.env.reset(**kwargs)
        obs, info = reset_out if isinstance(reset_out, tuple) else (reset_out, {})

        # 初始化延迟队列：全部填充默认动作（不喷火）
        self.action_queue.clear()
        for _ in range(self.action_delay_steps + 1):
            self.action_queue.append(Config.REAL_DEFAULT_ACTION)

        # 重置评分与阵风
        self.score_tracker.reset()
        self.gust_steps_left = 0
        self.gust_force = np.zeros(2, dtype=np.float32)

        # 对初始观测加噪声
        noisy_obs = self._add_sensor_noise(obs)
        return noisy_obs, dict(info)

    def step(self, action):
        """
        执行一步，按顺序：
            1. 从延迟队列取出延迟后的动作
            2. 施加阵风（如果有）
            3. 执行环境 step
            4. 更新评分追踪器
            5. 对下一帧观测加噪声
        """
        # 步骤 1：延迟链路 — 新动作入队，队头动作出队执行
        delayed_action = self._delayed_action(action)

        # 步骤 2：施加随机阵风
        self._apply_random_gust()

        # 步骤 3：执行环境步进
        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        # 步骤 4：更新评分
        self.score_tracker.update(
            obs=obs,
            executed_action=delayed_action,
            terminated=terminated,
            truncated=truncated,
        )

        # 步骤 5：对下一帧观测加传感器噪声
        noisy_obs = self._add_sensor_noise(obs)

        # 将 real 环境特有信息写入 info 字典
        info = dict(info)
        info["requested_action"] = int(np.asarray(action).reshape(-1)[0])      # 智能体发出的原始动作
        info["executed_action"] = int(np.asarray(delayed_action).reshape(-1)[0])  # 实际执行的动作
        info["gust_force"] = self.gust_force.astype(float).tolist()            # 当前阵风力
        info["gust_steps_left"] = int(self.gust_steps_left)                    # 阵风剩余帧数
        real_score = self.score_tracker.summary(done=terminated or truncated)
        info["real_score"] = real_score

        # 把 real 最终评分混入 terminal 奖励（评分目标对训练的反馈信号）。
        # 内层 LunarLanderRewardWrapper 计算奖励时还拿不到 real_score（见 env_factory 的包装顺序），
        # 所以这里在算完评分后再叠加。只有 episode 结束时 final_score 才非零，
        # 非终止帧 summary 返回 final_score=0，不影响每帧奖励。
        if terminated or truncated:
            final_score = float(real_score.get("final_score", 0.0))
            reward = float(reward) + Config.REWARD_SCALE * Config.REWARD_REAL_SCORE_WEIGHT * final_score

        return noisy_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # 传感器噪声
    # ------------------------------------------------------------------
    def _add_sensor_noise(self, obs):
        """
        在观测上叠加高斯噪声并 clip 到合法范围。

        noise ~ N(0, obs_noise_std²)

        注意：噪声加到 8 维观测上，包括：
            x, y（位置）— 范围 [-2.5, 2.5]
            vx, vy（速度）— 范围 [-10, 10]
            angle（角度）— 范围 [-2π, 2π]
            angular_vel（角速度）— 范围 [-10, 10]
            left_leg, right_leg（触地）— 范围 [0, 1]

        leg_contact 维度加了噪声可能变成负值或 >1，这是有意为之 —
        模拟接触传感器的不确定性。
        """
        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_noise_std <= 0.0:
            return obs
        noisy_obs = obs + self.rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
        return np.clip(noisy_obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    # ------------------------------------------------------------------
    # 控制延迟
    # ------------------------------------------------------------------
    def _delayed_action(self, action):
        """
        动作延迟管线。

        智能体每帧发出一个动作 → 进入 FIFO 队尾
        实际执行的是队头取出的 N 帧前的旧动作

        例如 delay_steps=8：
            t=0: 发出 action_0 → 执行 default_action（队列未填满）
            t=8: 发出 action_8 → 执行 action_0（8 帧前发出的）

        这意味着智能体看到的观测和它发出的动作之间存在 8 帧的错位。
        智能体必须学会"预测"8 帧后的状态来做出正确决策。
        """
        action = int(np.asarray(action).reshape(-1)[0])
        self.action_queue.append(action)
        return int(self.action_queue.popleft())

    # ------------------------------------------------------------------
    # 随机阵风
    # ------------------------------------------------------------------
    def _apply_random_gust(self):
        """
        随机阵风模拟。

        每帧检查：
            - 如果当前没有活跃阵风（gust_steps_left == 0）：
              以 REAL_GUST_PROBABILITY 概率触发新阵风
              随机采样阵风持续时间（8~32 帧）和力的大小
            - 如果有活跃阵风：
              对登月舱刚体中心施加阵风力
              gust_steps_left -= 1

        阵风力通过 Box2D 的 ApplyForceToCenter 施加，
        这是 Box2D 的物理接口，直接影响刚体运动。
        """
        lander = getattr(self.unwrapped, "lander", None)
        if lander is None:
            return   # 防御：某些渲染模式可能没有 lander 对象

        # 阵风到期 → 尝试触发新阵风
        if self.gust_steps_left <= 0 and self.rng.random() < Config.REAL_GUST_PROBABILITY:
            duration = self.rng.integers(
                Config.REAL_GUST_DURATION_MIN,
                Config.REAL_GUST_DURATION_MAX + 1,
            )
            self.gust_steps_left = int(duration)
            # 水平和垂直方向独立采样
            self.gust_force = np.array(
                [
                    self.rng.normal(0.0, Config.REAL_GUST_FORCE_X_STD),
                    self.rng.normal(0.0, Config.REAL_GUST_FORCE_Y_STD),
                ],
                dtype=np.float32,
            )

        # 施加阵风力
        if self.gust_steps_left > 0:
            lander.ApplyForceToCenter(
                (float(self.gust_force[0]), float(self.gust_force[1])),
                True,   # wake=True：唤醒刚体（如果它处于休眠状态）
            )
            self.gust_steps_left -= 1
        else:
            self.gust_force = np.zeros(2, dtype=np.float32)


# =============================================================================
# RealScoreTracker — real 环境最终评分计算
# =============================================================================
class RealScoreTracker:
    """
    逐局追踪并计算 real 环境最终评分。

    评分公式：
        final_score = (Wf * fuel + Wp * precision + Ws * stability) * completion

    其中：
        fuel_score ∈ [0, 100]：燃料消耗越少分越高
        precision_score ∈ [0, 100]：离目标中心越近分越高
        stability_score ∈ [0, 100]：飞行过程中角度/角速度/水平速度越小分越高
        completion ∈ {0, 1}：只有同时满足双腿着地、居中、姿态正才算完成

    权重（在 conf.py 中配置）：
        Wf = 0.20, Wp = 0.40, Ws = 0.40
    """

    def reset(self):
        """每局开始时重置所有统计量。"""
        self.steps = 0
        self.main_engine_count = 0     # 主发动机使用次数
        self.side_engine_count = 0     # 侧发动机使用次数
        self.final_abs_x = 1.0         # 最终 |x| 位置（初始化为 1.0 作为默认）
        self.final_abs_y = 1.0         # 最终 |y| 位置
        self.angle_abs_sum = 0.0       # 各帧 |angle| 累计和
        self.angular_vel_abs_sum = 0.0 # 各帧 |angular_vel| 累计和
        self.horizontal_vel_abs_sum = 0.0 # 各帧 |vx| 累计和
        self.completion_rate = 0.0     # 完成率：0 或 1

    def update(self, obs, executed_action, terminated=False, truncated=False):
        """
        每帧更新统计量。

        executed_action 而非 requested_action：
            — 评分基于实际执行的动作（延迟后的），不是智能体发出的原始动作
            — 这在延迟场景下很重要
        """
        state = np.asarray(obs, dtype=np.float32)
        action = int(np.asarray(executed_action).reshape(-1)[0])

        self.steps += 1

        # 统计发动机使用
        self.main_engine_count += int(action == 2)     # 主发动机
        self.side_engine_count += int(action in (1, 3))  # 侧发动机

        # 实时更新最终位置
        self.final_abs_x = abs(float(state[0]))
        self.final_abs_y = abs(float(state[1]))

        # 累计稳定性指标
        self.angle_abs_sum += abs(float(state[4]))         # |angle|
        self.angular_vel_abs_sum += abs(float(state[5]))   # |角速度|
        self.horizontal_vel_abs_sum += abs(float(state[2])) # |水平速度|

        # 终止时判断是否完成着陆
        if terminated:
            legs_contact = float(state[6] > 0.5 and state[7] > 0.5)  # 双腿着地
            centered = float(abs(state[0]) <= Config.REAL_COMPLETION_X_THRESHOLD)  # 水平居中
            level = float(abs(state[4]) <= Config.REAL_COMPLETION_ANGLE_THRESHOLD)  # 姿态水平
            self.completion_rate = legs_contact * centered * level
        elif truncated:
            self.completion_rate = 0.0  # 超时视为未完成

    def summary(self, done=False):
        """
        返回评分字典。

        参数 done: 是否在 episode 结束时调用。
                   只有 done=True 时才用 completion_rate 乘权重分。
                   中间帧返回 final_score=0。
        """
        fuel_score = self._fuel_score()
        precision_score = self._precision_score()
        stability_score = self._stability_score()

        # 加权综合分（不含完成率）
        weighted_score = (
            Config.REAL_SCORE_FUEL_WEIGHT * fuel_score
            + Config.REAL_SCORE_PRECISION_WEIGHT * precision_score
            + Config.REAL_SCORE_STABILITY_WEIGHT * stability_score
        )

        # 最终分数 = 加权分 × 完成率
        final_score = weighted_score * self.completion_rate if done else 0.0

        return {
            "fuel_score": fuel_score,
            "precision_score": precision_score,
            "stability_score": stability_score,
            "completion_rate": float(self.completion_rate),
            "weighted_score": float(weighted_score),
            "final_score": float(final_score),
        }

    def _fuel_score(self):
        """
        燃料消耗分 ∈ [0, 100]。

        燃油使用率 = (主发动机次数 + 0.25 * 侧发动机次数) / 总步数
        分 = 100 * (1 - 燃油使用率)

        思路：总步数越多 + 发动机使用越少 = 燃料效率越高
        侧发动机按 25% 折算因为推力小、油耗低。
        """
        if self.steps <= 0:
            return 100.0
        fuel_use = (
            self.main_engine_count
            + Config.REAL_SIDE_ENGINE_FUEL_RATIO * self.side_engine_count
        ) / self.steps
        return float(np.clip(100.0 * (1.0 - fuel_use), 0.0, 100.0))

    def _precision_score(self):
        """
        定位精准度分 ∈ [0, 100]。

        landing_error = sqrt(x² + y²)（距原点/目标中心的距离）
        分 = 100 * (1 - error / 1.5)

        error = 0 → score = 100（完美正中靶心）
        error ≥ 1.5 → score = 0
        """
        landing_error = np.sqrt(
            self.final_abs_x * self.final_abs_x
            + self.final_abs_y * self.final_abs_y
        )
        score = 100.0 * (1.0 - landing_error / max(Config.REAL_PRECISION_MAX_ERROR, 1e-6))
        return float(np.clip(score, 0.0, 100.0))

    def _stability_score(self):
        """
        机体平稳度分 ∈ [0, 100]。

        mean_instability = (1.0*|angle| + 0.5*|angular_vel| + 0.5*|vx|) / steps
        分 = 100 * (1 - mean_instability / 1.5)

        角度权重最高（1.0），因为倾斜最危险。
        角速度和水平速度权重各 0.5。
        整个过程都在考核，不是只看最后瞬间。
        """
        if self.steps <= 0:
            return 0.0
        mean_instability = (
            Config.REAL_STABILITY_ANGLE_WEIGHT * self.angle_abs_sum
            + Config.REAL_STABILITY_ANGULAR_VEL_WEIGHT * self.angular_vel_abs_sum
            + Config.REAL_STABILITY_HORIZONTAL_VEL_WEIGHT * self.horizontal_vel_abs_sum
        ) / self.steps
        score = 100.0 * (1.0 - mean_instability / max(Config.REAL_STABILITY_MAX_ERROR, 1e-6))
        return float(np.clip(score, 0.0, 100.0))
