# G1 搬箱 × Scaling-CRL × MotrixSim 训练指南

本文档说明如何在 UniLab 中训练 **G1 人形机器人 MotrixSim 搬箱任务**，算法为 **Scaling-CRL**（深度残差网络 + 对比式目标条件强化学习）。

原始算法参考：`/home/ubuntu22/data/scaling-crl/`（JAX/Flax 实现）。UniLab 侧为 PyTorch 移植，复用现有 off-policy 异构流水线（CPU 批量仿真 + GPU 策略学习）。

---

## 快速开始

### 前置依赖

```bash
# Motrix 后端
uv sync --extra motrix
```

### 一键训练（推荐）

```bash
uv run train --algo scaling_crl --task g1_box_placement --sim motrix
```

### 使用专用 Hydra 配置

```bash
uv run scripts/train_offpolicy.py \
  --config-path ../cfg/train \
  --config-name g1_box_motrix_scaling_crl
```

### 网络深度消融

`network_depth` 控制残差块数量（块数 = `network_depth // 4`）。

走 CLI 路由（`conf/offpolicy/config.yaml`）时，用 `algo.network_depth`，它会通过插值自动同步到 `actor_depth` / `critic_depth`：

```bash
# 深度 16（4 个残差块）
uv run train --algo scaling_crl --task g1_box_placement --sim motrix algo.network_depth=16

# 深度 64（16 个残差块）
uv run train --algo scaling_crl --task g1_box_placement --sim motrix algo.network_depth=64
```

走一键配置 `cfg/train/g1_box_motrix_scaling_crl.yaml` 时，可用顶层旋钮 `network_depth`：

```bash
uv run scripts/train_offpolicy.py \
  --config-path ../cfg/train --config-name g1_box_motrix_scaling_crl \
  network_depth=64
```

支持值：`4 / 8 / 16 / 32 / 64 / 256 / 1024`（与原 Scaling-CRL 一致，每 4 层为一个残差块）。

---

## 任务说明

### 场景布局

| 元素 | 配置 |
|------|------|
| 地面 | 平地，G1 站立于世界原点，面向正方向 |
| 箱子 | 正前方 **0.8 m**，尺寸 **0.2×0.2×0.3 m**，质量 **2 kg** |
| 目标平台 | 正前方 **1.5 m**，台面 **0.4×0.4 m**，高度 **0.6 m** |

场景 XML：`src/unilab/assets/robots/g1/scene_box_placement.xml`

### 任务目标

G1 完成全流程：**抓取地面箱子 → 步行至平台 → 平稳放置到台面**。

### 成功判定

单 episode 需同时满足（可通过 Hydra `env.success_criteria` 调整）：

| 条件 | 默认阈值 |
|------|----------|
| 箱子底部中心水平偏差 | ≤ 0.1 m |
| 箱子底部与台面高度差 | ≤ 0.05 m |
| 箱子倾斜角（与水平面） | ≤ 15° |
| 稳定保持 | ≥ 10 仿真步 |

### 奖励设计

稀疏 CRL 为主，辅以轻量塑形：

| 项 | 说明 |
|----|------|
| 成功放置 | +100 |
| 箱子-目标距离 | `-0.1 × L2` |
| 抓取滑移惩罚 | 末端与箱子相对位移过大时负奖励 |
| 姿态惩罚 | 躯干过度倾斜或跌倒时负奖励并终止 |

---

## Scaling-CRL 算法

### 核心逻辑

1. **Critic**：状态-动作编码器 φ 与目标编码器 ψ 输出 64 维嵌入，Q = -‖φ(s,a) - ψ(g)‖₂
2. **训练损失**：批内 InfoNCE 对比损失 + logsumexp 正则（系数 0.1）
3. **目标采样**：从同轨迹未来状态几何采样目标（`goal_relabeling.py`，对应原 `buffer.flatten_crl_fn`）
4. **Actor**：最大化 critic 输出（SAC 式熵正则，可选关闭）

### 网络结构

```
输入 → Dense(256) → LayerNorm → Swish
     → (network_depth // 4) × [Dense+LN+Swish ×4 + 残差连接]
     → Dense(64)
```

三套网络：**Actor**、**SA 编码器**、**Goal 编码器**，结构对齐原 `train.py`。

### Goal 空间（18 维，与观测分离）

```
[箱子 pos(3) + euler(3)] + [左末端 pos(3) + euler(3)] + [右末端 pos(3) + euler(3)]
```

观测分组：

| 组 | 维度 | 用途 |
|----|------|------|
| `obs` | 111 | state(93) ‖ goal(18)，供 Actor rollout |
| `critic` | 19 | goal(18) + episode_seed(1)，供 learner 重标注 |

---

## 默认超参数

| 参数 | 默认值 | 配置文件 |
|------|--------|----------|
| 学习率（actor/critic/α） | 3e-4 | `conf/offpolicy/algo/scaling_crl.yaml` |
| UTD 比例 | 1:40 | `updates_per_step: 40` |
| 折扣因子 γ | 0.99 | |
| Batch size | 512 | |
| 并行环境数 | 512 | |
| 回放缓冲区 | 10000 步 | `replay_buffer_n: 10000` |
| 隐藏层宽度 | 256 | |
| 嵌入维度 | 64 | |
| 评估间隔 | 10M 环境步 | `eval_interval_env_steps` |

---

## 代码结构

### 任务

| 路径 | 说明 |
|------|------|
| `src/unilab/envs/manipulation/g1/box_placement.py` | 任务主实现（注册名 `G1BoxPlacement`） |
| `src/unilab/tasks/g1_box_placement.py` | 别名入口 |
| `src/unilab/envs/manipulation/g1/success.py` | 成功判定模块 |

### 算法

| 路径 | 对应原代码 |
|------|-----------|
| `src/unilab/algos/torch/scaling_crl/networks.py` | `train.py` Actor / SA_encoder / G_encoder |
| `src/unilab/algos/torch/scaling_crl/losses.py` | `update_critic` InfoNCE |
| `src/unilab/algos/torch/scaling_crl/goal_relabeling.py` | `buffer.flatten_crl_fn` |
| `src/unilab/algos/torch/scaling_crl/learner.py` | 训练器 |
| `src/unilab/algorithms/scaling_crl/` | 别名包 |

### 配置

| 路径 | 说明 |
|------|------|
| `conf/offpolicy/algo/scaling_crl.yaml` | 算法超参 |
| `conf/offpolicy/task/scaling_crl/g1_box_placement/motrix.yaml` | Motrix 任务 owner YAML |
| `cfg/train/g1_box_motrix_scaling_crl.yaml` | 一键训练配置 |

### 训练入口

- CLI：`src/unilab/cli.py`（`scaling_crl` 已注册到 off-policy 路由）
- 脚本：`scripts/train_offpolicy.py` → `DoubleBufferOffPolicyRunner`

---

## 异构架构

```
┌─────────────────────┐     ┌──────────────────────┐
│  Collector (CPU)    │     │  Learner (GPU)       │
│  MotrixSim ×512 env │────▶│  ScalingCRLLearner   │
│  ScalingCRLActor    │     │  InfoNCE + Actor更新  │
└─────────┬───────────┘     └──────────┬───────────┘
          │  ReplayBuffer (共享内存)      │
          │  CPUPinnedDoubleBuffer H2D   │
          └──────────────────────────────┘
```

- 采集与训练流水重叠，不阻塞
- 权重通过 `SharedWeightSync` 同步到 collector
- 日志 / checkpoint 复用 UniLab off-policy 能力

---

## 评估指标

训练日志（TensorBoard / WandB）包含：

| 指标 | 来源 |
|------|------|
| `success_rate` | 每步成功帧比例 |
| `episode_success` | episode 成功事件 |
| `success_episode_length` | 成功 episode 平均步长 |
| `critic_loss` / `actor_loss` | learner 更新 |
| `mean_ep100` | 近 100 episode 平均奖励 |

### 查看 TensorBoard

训练日志默认写入 `logs/scaling_crl/G1BoxPlacement/<时间戳>_motrix/`：

```bash
# 查看某次任务的全部 run
uv run tensorboard --logdir logs/scaling_crl/G1BoxPlacement

# 或只看全部 scaling_crl 任务
uv run tensorboard --logdir logs/scaling_crl

# 远程服务器：绑定 0.0.0.0 后本地浏览器访问 http://<server-ip>:6006
uv run tensorboard --logdir logs/scaling_crl/G1BoxPlacement --host 0.0.0.0 --port 6006
```

---

## 常用 Hydra 覆盖

```bash
# 调整成功阈值
uv run train --algo scaling_crl --task g1_box_placement --sim motrix \
  env.success_criteria.horizontal_tolerance_m=0.08

# 减小并行规模（调试）
uv run train --algo scaling_crl --task g1_box_placement --sim motrix \
  algo.num_envs=64 algo.batch_size=256

# 仅训练不播放
uv run train --algo scaling_crl --task g1_box_placement --sim motrix \
  training.no_play=true

# 从 checkpoint 恢复
uv run train --algo scaling_crl --task g1_box_placement --sim motrix \
  training.resume=true algo.load_run=<run_dir_name>
```

---

## 测试

```bash
uv run pytest tests/algos/test_scaling_crl.py \
               tests/envs/manipulation/test_g1_box_placement.py -q
```

---

## 与 G1 Box Tracking 的区别

| | G1 Box Tracking | G1 Box Placement |
|--|-----------------|------------------|
| 范式 | 运动模仿（NPZ 参考轨迹） | 目标条件 RL（CRL 自监督） |
| 算法 | PPO / FlashSAC 等 | Scaling-CRL |
| 奖励 | 轨迹跟踪指数核 | 稀疏成功 + 轻量塑形 |
| 目标 | 跟踪参考动作 | 箱子放置到平台 |
