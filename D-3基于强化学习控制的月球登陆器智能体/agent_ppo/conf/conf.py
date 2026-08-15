from __future__ import annotations

from pathlib import Path


# =============================================================================
# Config — LunarLander PPO 全局配置中心
# =============================================================================
# 本文件是整个项目的参数入口，所有模块都从这里读取配置。
# 二次开发时，优先在这里调整参数，避免分散在各处硬编码。
#
# 调参优先级建议：
#   第一层（影响最大）：奖励权重、real 环境噪声/延迟参数
#   第二层（中等影响）：PPO 超参 (learning_rate, gamma, ent_coef)
#   第三层（精细调节）：网络结构、训练步数、eval 频率
# =============================================================================

class Config:
    # ------------------------------------------------------------------
    # 路径与项目标识
    # ------------------------------------------------------------------
    ROOT_DIR = Path(__file__).resolve().parents[2]   # rl-LunarLander/

    ALGO = "ppo"                                     # 算法名称，用于日志路径
    ENV_ID = "LunarLander-v3"                        # Gymnasium 环境 ID
    ORG = "sb3"                                      # 组织标识（保留，未使用）
    LOG_FOLDER = ROOT_DIR / "logs"                   # 模型和评估日志根目录

    # ------------------------------------------------------------------
    # PPO 训练基础参数
    # ------------------------------------------------------------------
    POLICY = "MlpPolicy"                             # 策略类型：MLP 多层感知机
                                                     #   MlpPolicy 适合 8 维观测这种低维向量输入
                                                     #   如果改用图像输入，需换成 CnnPolicy

    # 环境包装器列表 — 在环境创建时按顺序套上
    # RewardWrapper 必须放在最内层（直接包装 gym 环境）
    # real 环境的 RealLunarLanderWrapper 在 env_factory 中额外套在 RewardWrapper 外面
    ENV_WRAPPERS = [
        "agent_ppo.feature.reward_process.LunarLanderRewardWrapper",
    ]

    # ---------- 训练量 ----------
    N_TIMESTEPS = 2_000_000      # 总训练步数（跨所有并行环境累计）
                                  #   LunarLander 通常需要 500K~1M 步才能稳定着陆
                                  #   real 环境因为有噪声/延迟，可能需要增加到 2M~5M
    N_ENVS = 32                   # 并行环境数
                                  #   越大采样效率越高，但内存占用也越大
                                  #   CPU 训练建议 8~16，GPU 可以更多

    # ---------- PPO 超参 ----------
    N_STEPS = 1024                # 每次策略更新前收集的步数（每个环境）
                                  #   1024 是 LunarLander 的常用值
                                  #   太小 → 方差大，太大 → 样本利用率低

    BATCH_SIZE = 128              # 小批量大小
                                  #   每次从 N_STEPS*N_ENVS 条经验中随机抽取 BATCH_SIZE 条做梯度更新

    N_EPOCHS = 10                  # 每轮数据重复使用的 epoch 数
                                  #   PPO 允许多次复用同一批数据，但不能太多（策略偏移）

    LEARNING_RATE = 0.0003        # Adam 优化器学习率（初始值）
                                  #   训练时线性退火到 0（见 algorithm_ppo.py 的 LinearSchedule）
                                  #   3e-4 是 PPO 的经典默认值
                                  #   如果训练不稳定（real 环境噪声大），可降至 1e-4

    GAMMA = 0.99              # 折扣因子
                                  #   接近 1 表示重视远期回报
                                  #   LunarLander 着陆需要长时序规划，所以 gamma 设得很高

    GAE_LAMBDA = 0.98             # GAE (Generalized Advantage Estimation) λ
                                  #   接近 1 → 低偏差高方差，接近 0 → 高偏差低方差
                                  #   0.98 适合需要精确价值估计的任务

    CLIP_RANGE = 0.2              # PPO clip 范围
                                  #   限制新旧策略之间的概率比在 [1-ε, 1+ε]
                                  #   0.2 是经典值，训练不稳时可降到 0.1

    ENT_COEF = 0.01               # 熵正则化系数（初始值）
                                  #   训练时由 EntropyDecayCallback 线性退火到 0（见 train_workflow.py）
                                  #   前期鼓励探索，末期收敛到确定性策略
                                  #   real 环境噪声大，可能需要增加到 0.02~0.05

    VF_COEF = 0.5                 # 价值函数损失权重
                                  #   控制价值网络训练在总损失中的占比

    MAX_GRAD_NORM = 0.5           # 梯度裁剪阈值
                                  #   防止梯度爆炸，0.5 是常用值

    # ---------- 观测/回报归一化 ----------
    NORMALIZE = False             # 是否启用 VecNormalize
                                  #   对观测和回报做运行均值方差归一化
                                  #   LunarLander 的观测值范围已知且稳定，
                                  #   一般不需要归一化。如果启用，测评时必须加载 normalizer。
    NORM_OBS = False
    NORM_REWARD = False

    # ---------- 随机种子 ----------
    TRAIN_SEED = 0                # 训练环境种子（固定以保证可复现）
    EVAL_SEED = 10_000            # 评估环境种子（与训练不同，测试泛化能力）

    # ---------- 评估 ----------
    EVAL_FREQ_STEPS = 25_000      # 每隔多少步做一次评估
                                  #   实际频率 = EVAL_FREQ_STEPS // num_envs
                                  #   16 envs → 每 1562 步评估一次
    N_EVAL_EPISODES = 20          # 每次评估跑多少局
    EVAL_SUCCESS_REWARD = 200.0   # 通关阈值：一局累计回报 ≥ 200 即视为成功着陆

    # ==================================================================
    # Real 环境参数 — 二次开发的核心调参区
    # ==================================================================
    # 这些参数模拟真实世界中的不完美条件：
    #   - 传感器噪声：GPS/IMU 读数不准
    #   - 控制延迟：指令下达到执行有时间差
    #   - 随机阵风：外部扰动
    #
    # 调参思路：
    #   噪声/延迟越大 → 任务越难 → 需要更强的泛化能力和更大的网络
    #   建议先用较小噪声验证算法能收敛，再逐步增加难度

    # --- 传感器噪声 ---
    REAL_OBS_NOISE_STD = 0.10     # 高斯噪声标准差
                                   #   加到 8 维观测的每一维上
                                   #   观测值范围约 [-2.5, 2.5]（位置）和 [-10, 10]（速度）
                                   #   0.10 的噪声约为位置范围的 2%，速度范围的 0.5%
                                   #   增大 → 定位更难，需要智能体学会滤波

    # --- 控制延迟 ---
    REAL_ACTION_DELAY_STEPS = 8    # 动作延迟帧数
                                   #   8 帧 × 50Hz = 160ms 延迟
                                   #   这模拟真实推进器的点火延迟
                                   #   越大 → 智能体必须学会"预判"，不能等看到偏差再反应
    REAL_DEFAULT_ACTION = 0        # 延迟期间的默认动作（0 = 不喷火）
                                   #   在 pipeline 填充的帧中执行此动作

    REAL_NOISE_SEED_OFFSET = 2026  # 噪声种子偏移量
                                   #   避免 real 环境的随机数生成器与训练环境相同

    # --- 随机阵风 ---
    REAL_GUST_PROBABILITY = 0.18   # 每帧触发阵风的概率
                                   #   0.18 意味着平均每 5~6 帧可能遇到一次阵风检查
                                   #   实际阵风持续 8~32 帧
    REAL_GUST_FORCE_X_STD = 3.0   # 阵风水平力的标准差（牛顿）
    REAL_GUST_FORCE_Y_STD = 1.0   # 阵风垂直力的标准差（牛顿）
                                   #   水平风比垂直风大，因为现实中侧风影响更显著
    REAL_GUST_DURATION_MIN = 8     # 阵风最短持续帧数
    REAL_GUST_DURATION_MAX = 32    # 阵风最长持续帧数

    # ==================================================================
    # Real 环境最终评分权重 — 二次开发的核心调参区
    # ==================================================================
    # 最终公式：final_score = (Wf*fuel + Wp*precision + Ws*stability) * completion
    # 其中 completion ∈ {0, 1}：必须同时满足双脚着地、x 偏移 < 0.25、角度 < 0.35 rad
    #
    # 调参思路：
    #   当前权重 (0.2, 0.4, 0.4) 偏重定位精度和稳定性
    #   如果希望节省燃料，增大 fuel 权重
    #   如果只关心能不能着陆（不在乎精度），降低 precision/stability 权重
    REAL_SCORE_FUEL_WEIGHT = 0.20       # 燃料消耗分权重
    REAL_SCORE_PRECISION_WEIGHT = 0.40  # 定位精准度分权重
    REAL_SCORE_STABILITY_WEIGHT = 0.40  # 机体平稳度分权重

    # --- 燃料分计算参数 ---
    REAL_SIDE_ENGINE_FUEL_RATIO = 0.25  # 侧发动机燃料消耗折算比
                                         #   侧发动机每次消耗按主发动机的 25% 计算
                                         #   因为侧发动机推力小，燃料消耗也少

    # --- 定位精准度分计算参数 ---
    REAL_PRECISION_MAX_ERROR = 1.5      # 定位误差归一化上限（米）
                                         #   离目标 > 1.5m → precision_score = 0
                                         #   完美着陆中心 → precision_score = 100

    # --- 机体平稳度分计算参数 ---
    REAL_STABILITY_MAX_ERROR = 1.5      # 不稳定度归一化上限
    REAL_STABILITY_ANGLE_WEIGHT = 1.0        # 角度在稳定性分中的权重
    REAL_STABILITY_ANGULAR_VEL_WEIGHT = 0.5  # 角速度权重
    REAL_STABILITY_HORIZONTAL_VEL_WEIGHT = 0.5  # 水平速度权重
    # 当前：稳定性 = 角度(1.0) + 角速度(0.5) + 水平速度(0.5)
    # 也就是角度最重要，其次是不能晃得太厉害、不能水平飘移

    # --- 完成条件判断阈值 ---
    REAL_COMPLETION_X_THRESHOLD = 0.25   # 水平位置偏差阈值（米）
                                          #   |x| ≤ 0.25 才判定为"居中着陆"
    REAL_COMPLETION_ANGLE_THRESHOLD = 0.35  # 角度阈值（弧度 ≈ 20°）
                                             #   |angle| ≤ 0.35 才判定为"姿态正确"

    # ==================================================================
    # 奖励函数参数 — 二次开发的核心调参区
    # ==================================================================
    # 奖励 = shaping_delta + engine_cost，terminal 时替换为 terminal_reward
    #
    # shaping_delta = 当前帧 shaping - 上一帧 shaping
    # shaping = 距离惩罚 + 速度惩罚 + 角度惩罚 + 腿接触奖励
    #
    # 调参思路：
    #   当前权重接近 Gymnasium 原始 LunarLander 奖励
    #   对 real 环境，可能需要增大 landing reward 来鼓励完成着陆
    #   或降低 crash penalty 避免智能体过于保守（怕坠毁所以不敢靠近地面）

    REWARD_DISTANCE_WEIGHT = -100.0       # 距离惩罚系数
                                           #   shaping 项: -100 * sqrt(x² + y²)
                                           #   越靠近着陆点（原点），惩罚越小
    REWARD_VELOCITY_WEIGHT = -100.0       # 速度惩罚系数
                                           #   shaping 项: -100 * sqrt(vx² + vy²)
                                           #   速度越快惩罚越大，鼓励缓慢下降
    REWARD_ANGLE_WEIGHT = -100.0          # 角度惩罚系数
                                           #   shaping 项: -100 * |angle|
                                           #   倾斜越大惩罚越大，鼓励保持竖直
    REWARD_LEG_CONTACT_BONUS = 10.0       # 腿接触奖励
                                           #   每条腿接触地面给予 +10 shaping 奖励

    REWARD_MAIN_ENGINE_COST = 0.90        # 主发动机燃料惩罚
                                           #   每次开主发动机 → -0.90
                                           #   主发动机推力大、耗油多 → 惩罚更重
    REWARD_SIDE_ENGINE_COST = 0.30        # 侧发动机燃料惩罚
                                           #   每次开侧发动机 → -0.30
                                           #   侧发动机推力小、耗油少 → 惩罚轻

    REWARD_CRASH = -100.0                 # 坠毁惩罚
                                           #   当环境返回 reward ≤ -100 时触发
    REWARD_LANDING = 100.0                # 成功着陆基础奖励
                                           #   当环境返回 reward ≥ +100 时触发
                                           #   这是 terminal 奖励，替换掉整帧的 shaping + fuel
    REWARD_LANDING_PRECISION_BONUS = 100.0 # 着陆定位精度奖励上限
                                           #   在基础 REWARD_LANDING 之上，根据 |x| 位置额外加分
                                           #   公式：bonus = BONUS × (1 - |x| / X_RANGE)
                                           #   |x|=0    → 100 + 100  = 200
                                           #   |x|=0.15 → 100 + 50   = 150
                                           #   |x|≥0.3  → 100 + 0    = 100
    REWARD_LANDING_X_RANGE = 0.3           # 定位奖励衰减范围（米）
                                           #   |x| 超过此值则不再有精度加分

    REWARD_SCALE = 1.0                    # 奖励缩放因子
                                           #   final_reward = REWARD_SCALE * computed + REWARD_BIAS
    REWARD_BIAS = 0.0                     # 奖励偏移
    REWARD_REAL_SCORE_WEIGHT = 0.9  # real_score 在 terminal 奖励中的权重
                                    #   混入位置在 RealLunarLanderWrapper.step()（agent_ppo/feature/real_env.py）
                                    #   terminal_reward = 基础(±100+精度加成) + weight * final_score
                                    #   final_score ∈ [0, 100]，weight=0.3 → 最多额外 +30
                                    #   terminal 最大 = 100 + 100 + 30 = 230
                                    #   设为 0 则完全不引入 real_score

    # ==================================================================
    # 观测/动作空间定义
    # ==================================================================
    OBS_DIM = 8                            # 观测维度
                                           #   [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
    STATE_SHAPE = (OBS_DIM,)
    STATE_DIM = OBS_DIM
    ACTION_NUM = 4                         # 离散动作数：0=不喷火, 1=左, 2=主, 3=右
    VALUE_NUM = 1                          # 价值输出维度（标量）

    # ==================================================================
    # 网络结构参数 — 二次开发的调参区
    # ==================================================================
    # 当前结构：Actor [128, 128] + Critic [128, 128]，ReLU 激活
    # 这是 SB3 MlpPolicy 的经典配置，适合 8→4 的简单映射
    #
    # 调参思路：
    #   real 环境噪声大 → 可能需要更深的网络（如 [128, 128] 或 [256, 128, 64]）
    #   更大的网络 → 更强的表达能力，但训练更慢、更容易过拟合
    #   也可以尝试 ReLU/ELU 替代 Tanh（Tanh 梯度在深层容易饱和）
    ACTOR_HIDDEN_LAYERS = [128, 128]       # Actor 隐藏层
    CRITIC_HIDDEN_LAYERS = [128, 128]      # Critic 隐藏层
    ACTIVATION_FN = "nn.ReLU"              # 激活函数：Tanh / ReLU / ELU / LeakyReLU
    ORTHO_INIT = True                      # 正交初始化（PPO 推荐开启）
