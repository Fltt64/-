"""
训练工作流 — LunarLander PPO 训练的完整编排。

================================================================================
流程概览

    1. 创建训练环境（VecEnv，多进程并行）
    2. 创建评估环境（单进程，独立随机种子）
    3. 构建 PPO 算法实例
    4. 设置 EvalCallback（定期评估 + 自动保存最优模型）
    5. agent.learn() 训练主循环
    6. 保存最终模型

================================================================================
关键设计决策

    - 训练环境用 VecEnv（多个独立环境并行采样，提高效率）
    - 评估环境用单个环境（保证评估结果一致可比较）
    - 最优模型基于 mean_reward 自动保存（best_model.zip）
    - EvalCallback 同时承担 TensorBoard 日志、评估曲线、最优模型保存
    - 评估频率 = EVAL_FREQ_STEPS // num_envs（自动适配并行度）

================================================================================
二次开发注意点

    - eval_freq 是步数阈值（不是 episode 数），并行环境越多评估越频繁
    - best_model.zip 覆盖式保存，只保留目前最优的一个
    - 如果长时间评估没有提升，可以增大 EVAL_FREQ_STEPS 减少评估开销
    - VecNormalize 的状态需要在训练/测评间保持同步，否则测评结果会错误
================================================================================
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from agent_ppo.conf.conf import Config
from agent_ppo.workflow.env_factory import BASE_ENV_MODE, ENV_MODES, make_lunarlander_env, run_name


# =============================================================================
# SaveBestVecNormalizeCallback — 同步保存 VecNormalize 统计量
# =============================================================================
class SaveBestVecNormalizeCallback(BaseCallback):
    """
    当选到新的最优模型时，同时保存当前的观测归一化统计量。

    为什么需要这个：
        VecNormalize 维护 obs_rms（运行均值/方差），在训练时不断更新。
        如果只保存模型权重而不保存归一化状态，测评时观测分布会不匹配。
    """

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        """被 EvalCallback 在每个 eval 步调用。"""
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(self.save_path))
        return True


# =============================================================================
# EntropyDecayCallback — 熵退火
# =============================================================================
class EntropyDecayCallback(BaseCallback):
    """
    线性退火 ent_coef。

    SB3 2.9 的 ent_coef 不支持 schedule（train() 里直接 self.ent_coef * entropy_loss），
    所以用回调在每个 rollout 后手动把 ent_coef 从 start 线性衰减到 end。

    学习率 LR 走的是 LinearSchedule（见 algorithm_ppo.py），但 ent_coef 没有等价机制，
    只能走回调。num_timesteps 在每次 learn() 时会重置，课程学习（--init-model）也正确重新退火。
    """

    def __init__(self, start: float, end: float, total_timesteps: int):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.total = max(int(total_timesteps), 1)

    def _on_step(self) -> bool:
        frac = min(max(self.num_timesteps / self.total, 0.0), 1.0)
        self.model.ent_coef = self.start + (self.end - self.start) * frac
        return True


# =============================================================================
# _next_run_id — 实验序号管理
# =============================================================================
def _next_run_id(name: str) -> int:
    """
    自动递增实验序号。

    例如日志目录下已有 LunarLander-v3_1、LunarLander-v3_2，
    则下一个 run_id = 3。

    这保证了每次训练都有独立的存储目录，不会互相覆盖。
    """
    algo_dir = Config.LOG_FOLDER / Config.ALGO
    algo_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{name}_"
    run_ids = []
    for item in algo_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            suffix = item.name[len(prefix):]
            if suffix.isdigit():
                run_ids.append(int(suffix))
    return max(run_ids, default=0) + 1


# =============================================================================
# workflow — 训练主流程
# =============================================================================
def workflow(
    timesteps: int | None = None,    # 覆盖 Config.N_TIMESTEPS
    n_envs: int | None = None,       # 覆盖 Config.N_ENVS
    device: str = "auto",            # "auto" / "cpu" / "cuda"
    env_mode: str = BASE_ENV_MODE,   # "base" 或 "real"
    init_model_path: str | Path | None = None,  # 课程学习：从该模型继续训练（如 base 的 best_model.zip）
) -> None:
    """
    LunarLander PPO 训练主流程。

    参数:
        timesteps: 总训练步数，None 则使用 Config 默认值
        n_envs: 并行环境数，None 则使用 Config 默认值
        device: 计算设备
        env_mode: 环境模式 — "base"（原始）或 "real"（噪声+延迟+阵风）
        init_model_path: 课程学习用。给定一个已训练好的模型路径（如 base 的
            best_model.zip），则从该模型权重继续训练，而不是从头开始。
            典型用法：先 train_base.py 收敛，再 train_real.py --init-model 该模型。
    """
    # ---- 参数解析 ----
    if env_mode not in ENV_MODES:
        raise ValueError(f"Unsupported env_mode: {env_mode}. Expected one of {ENV_MODES}")
    total_timesteps = int(timesteps or Config.N_TIMESTEPS)
    num_envs = int(n_envs or Config.N_ENVS)
    target_env_id = Config.ENV_ID
    target_run_name = run_name(env_mode, target_env_id)
    # target_run_name: base → "LunarLander-v3", real → "LunarLander-v3-real"

    # ---- 创建运行目录 ----
    run_id = _next_run_id(target_run_name)
    run_dir = Config.LOG_FOLDER / Config.ALGO / f"{target_run_name}_{run_id}"
    model_stats_dir = run_dir / target_run_name
    model_stats_dir.mkdir(parents=True, exist_ok=True)
    # run_dir:        logs/ppo/LunarLander-v3_7/
    # model_stats_dir: logs/ppo/LunarLander-v3_7/LunarLander-v3/

    # ---- 创建训练环境（并行）----
    # make_vec_env 将单个 env 工厂函数复制为 N 个并行环境
    env = make_vec_env(
        lambda: make_lunarlander_env(env_mode),
        n_envs=num_envs,
        seed=Config.TRAIN_SEED,
    )
    if Config.NORMALIZE:
        env = VecNormalize(env, norm_obs=Config.NORM_OBS, norm_reward=Config.NORM_REWARD)

    # ---- 创建评估环境（单进程）----
    eval_env = make_vec_env(
        lambda: make_lunarlander_env(env_mode),
        n_envs=1,   # 评估用单环境，保证结果确定性
        seed=Config.EVAL_SEED,
    )
    if Config.NORMALIZE:
        # training=False：评估时不更新 running stats
        eval_env = VecNormalize(
            eval_env,
            norm_obs=Config.NORM_OBS,
            norm_reward=Config.NORM_REWARD,
            training=False,
        )

    # ---- 构建 PPO 算法 ----
    agent = AlgorithmPPO(env, device=device, init_model_path=init_model_path)

    # ---- 设置 EvalCallback ----
    # EvalCallback 在每个 eval_freq 步自动：
    #   1. 用当前策略在 eval_env 中跑 N_EVAL_EPISODES 局
    #   2. 记录平均回报到 TensorBoard
    #   3. 如果 mean_reward 创新高 → 保存 best_model.zip
    #   4. 保存 evaluations.npz（评估曲线数据）
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),     # best_model.zip 保存位置
        log_path=str(run_dir),                  # evaluations.npz 保存位置
        eval_freq=max(Config.EVAL_FREQ_STEPS // num_envs, 1),
                                               # 自动适配并行度
        n_eval_episodes=Config.N_EVAL_EPISODES,
        deterministic=True,                    # 评估用确定性策略（不采样）
        callback_on_new_best=SaveBestVecNormalizeCallback(
            model_stats_dir / "best_vecnormalize.pkl"
        ),
    )

    # ---- 熵退火回调：ent_coef 从 Config.ENT_COEF 线性退到 0 ----
    entropy_callback = EntropyDecayCallback(
        start=Config.ENT_COEF,
        end=0.005,
        total_timesteps=total_timesteps,
    )

    # ---- 开始训练 ----
    agent.learn(
        total_timesteps,
        callback=CallbackList([entropy_callback, eval_callback]),
        progress_bar=True,
    )

    # ---- 保存最终模型 ----
    model_path = run_dir / f"{target_run_name}.zip"
    agent.save(model_path)
    if Config.NORMALIZE:
        env.save(str(model_stats_dir / "vecnormalize.pkl"))

    # ---- 清理 ----
    env.close()
    eval_env.close()

    # ---- 输出结果 ----
    print(f"saved: {model_path.resolve()}")
    print(f"env_id: {target_env_id}")
    print(f"env_mode: {env_mode}")
    print(f"device: {device}")
    print(f"exp_id: {run_id}")
    if init_model_path is not None:
        print(f"init_model: {Path(init_model_path).resolve()}")
