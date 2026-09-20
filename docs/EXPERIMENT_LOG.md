# 完整实验记录

本文档记录项目从基准确认、视觉消融、归一化诊断到 PPO 优化与失败尝试的完整过程。除特别标注外，训练预算均为 10M environment steps，训练 seed 为 1，最终评估使用 seeds 101/202/303、每个 100 episodes。

## 1. 统一设置

| 配置 | 数值 |
|---|---|
| Environment | `StackCube-v1` |
| Robot | `panda_wristcam` |
| Cameras | base + hand, 128×128 RGB |
| Simulator | SAPIEN / PhysX CUDA |
| Parallel envs | 512 |
| Rollout length | 50 |
| PPO minibatches / epochs | 32 / 4 |
| Learning rate | 3e-4 |
| Gamma / GAE lambda | 0.8 / 0.9 |
| PPO clip / target KL | 0.2 / 0.2 |
| Reward | normalized dense |
| Action mode | `pd_joint_delta_pos` |

## 2. 参考基准与信息边界

| 实验 | Actor / Critic | Once | End | Return | 说明 |
|---|---|---:|---:|---:|---|
| S0-10M | 完整 48-D state | 54.3% | 32.7% | 32.22 | 无图像 state PPO |
| S0-25M | 完整 48-D state | 67.3% | 46.3% | 34.20 | 25M、1024 env，非同预算参考 |
| V0 | RGB + 完整 48-D state | 73.3% | 63.0% | 34.08 | Actor 读取物体真值，只作上限参考 |
| P0 | RGB only | 44.7% | 30.0% | 32.50 | 不使用 proprio |
| B0 | RGB + 25-D proprio | 21.3% | 4.0% | 31.85 | Actor/Critic 都不能读取特权状态 |

早期最重要的发现是：V0 的高性能不能被称为纯视觉，因为完整 state 直接包含 cube pose 和相对目标位置；B0 虽然部署信息边界正确，但 raw proprio 与图像特征尺度失配，性能反而低于 P0。

## 3. B1–B4 单变量视觉链

| 实验 | 唯一新增项 | Once | End | Return |
|---|---|---:|---:|---:|
| B0 | RGB + proprio | 21.3% | 4.0% | 31.85 |
| B1 | 特权 Critic | 79.0% | 46.7% | 35.31 |
| B2 | 双相机独立 CNN | 65.3% | 16.3% | 32.46 |
| B3 | 4-frame stack | 61.3% | 19.0% | 32.63 |
| B4 | 23-D privileged-state auxiliary head | 54.7% | 35.7% | 32.21 |

B1 表明 privileged Critic 能在不泄漏部署信息的情况下显著帮助视觉 Actor。B2–B4 的 raw 结果并非单调增加，这促使项目转向输入尺度诊断，而不是继续盲目堆模块。

## 4. Normalization 链与 B1 分支诊断

RGB 始终除以 255。Norm 版本只增加逐维 running mean/std，并裁剪到 `[-5,5]`；统计量作为模型 buffer 保存，评估时冻结。

| 实验 | Once | End | Return | 相对 raw 的观察 |
|---|---:|---:|---:|---|
| B0-Norm | 73.0% | 63.3% | 35.25 | +51.7pp once / +59.3pp end |
| B1-Norm | 68.3% | 32.0% | 33.58 | 低于 B1 raw |
| B2-Norm | 65.7% | 34.0% | 33.04 | once 近似不变，end 改善 |
| B3-Norm | 62.0% | 41.0% | 32.87 | once 近似不变，end +22pp |
| B4-Norm | 84.0% | 68.0% | 37.57 | 辅助任务与 normalization 显著协同 |
| B1-ActorNorm | 49.3% | 41.0% | 31.99 | 只归一化 Actor proprio；更保守但到达率下降 |
| B1-CriticNorm | — | — | — | 已实现和启动诊断，但最终数值未保留在本地归档 |

这组实验说明 normalization 不增加任务信息，也不保证必然变好。它改善有限网络和固定优化预算下的条件数，但会与共享表征、特权 Critic、探索方差和训练 seed 发生交互。

B4-Norm 的关键交叉对照：

| 对照 | Once 变化 | End 变化 | 解释 |
|---|---:|---:|---|
| B3 → B3-Norm | +0.7pp | +22.0pp | 归一化主要改善精细控制和保持 |
| B3-Norm → B4-Norm | +22.0pp | +27.0pp | 归一化后的辅助几何监督显著有效 |
| B4 → B4-Norm | +29.3pp | +32.3pp | 两者存在明显协同 |

## 5. B4-Norm 上的 PPO 优化

所有实验只修改一项，网络、reward、10M steps、gamma=0.8 和 seed 1 不变。

| 实验 | 唯一变化 | Once | End | Return | 结论 |
|---|---|---:|---:|---:|---|
| B4-Norm | 对照 | 84.0% | 68.0% | 37.57 | 单 seed 主基线 |
| AnnealLR | LR 3e-4 线性降至 0 | 83.7% | 65.0% | 37.30 | 基本持平 |
| Beta-PPO | Gaussian → Beta policy | 53.3% | 1.3% | 31.87 | 明显退化 |
| TCAPS | temporal coefficient 0.1 | 89.3% | 67.0% | 38.35 | once +5.3pp，end 未提高 |
| Actor-DrAC Photo | brightness/contrast policy KL | 0.0% | 0.0% | 0.49 | 完全失败 |

实现验证包括默认 Gaussian 路径逐元素等价、Beta bounded action 和正确变换密度、CAPS reset mask、DrAC RGB-only 隔离、checkpoint roundtrip 与 32-env smoke。

## 6. NoEarlyReset：主结果

原配置训练时 `partial_reset=True`，达到 success 后立即 reset；评估时 `eval_partial_reset=False`，必须继续到第 50 步。训练从未要求策略学习成功后的释放、撤离和保持。

NoEarlyReset 的唯一变化：

```text
B4-Norm:              partial_reset=True
B4-Norm-NoEarlyReset: partial_reset=False
```

网络、reward、PPO 超参数和总步数均不变。

### 三训练 seed

| Train seed | Once | End | Retention | Return |
|---:|---:|---:|---:|---:|
| 1 | 88.7% | 82.3% | 92.9% | 38.01 |
| 2 | 87.0% | 78.0% | 89.7% | 36.80 |
| 3 | 79.7% | 71.0% | 89.1% | 35.65 |
| Mean ± std | 85.1% ± 4.8pp | 77.1% ± 5.7pp | 90.6% aggregate | 36.82 ± 1.18 |

与 B4-Norm 配对比较：

| 方法 | Once | End | Return |
|---|---:|---:|---:|
| B4-Norm | 65.7% ± 23.9pp | 40.1% ± 34.6pp | 34.99 ± 2.57 |
| NoEarlyReset | 85.1% ± 4.8pp | 77.1% ± 5.7pp | 36.82 ± 1.18 |

同一 train seed 2 下，B4-Norm 只有 38.7%/1.3%，NoEarlyReset 达到 87.0%/78.0%。这排除了“seed 2 天生困难”的简单解释，也说明 termination mismatch 不仅影响保持，还会放大 on-policy 训练早期数据分布差异。

## 7. TCAPS 与 NoEarlyReset 组合

| 方法 | Once | End | Retention | Return |
|---|---:|---:|---:|---:|
| TCAPS | 89.3% | 67.0% | 75.0% | 38.35 |
| NoEarlyReset | 88.7% | 82.3% | 92.9% | 38.01 |
| NoEarlyReset + TCAPS | 87.3% | 77.3% | 88.5% | 37.49 |
| NoEarlyReset + pre-success TCAPS | 86.0% | 79.3% | 92.2% | 37.55 |

全程组合并未叠加增益。TCAPS 的平滑先验帮助抓取搬运，但限制成功后的快速释放和纠错。将 TCAPS 在首次成功后关闭，使 end 从 77.3% 恢复到 79.3%，验证了阶段冲突假设；但共享 Actor 参数仍然承受成功前的 TCAPS 梯度，因此未超过纯 NoEarlyReset。

## 8. 其他失败尝试

### gSDE on B0-Norm

| Control | Once | End | Return |
|---|---:|---:|---:|
| B0-Norm | 73.0% | 63.3% | 35.25 |
| B0-Norm + gSDE | 0.0% | 0.0% | 24.62 |

实现遵循 PPO gSDE 的 feature-by-action noise matrix，并匹配 B0-Norm 的初始边际 action std。代码和数学测试全部通过，但训练中 Actor latent 尺度持续增大：有效 action std 约从 0.58 增至 5.62，continuity cost 从 18.8 增至 1827，entropy 从 11.1 增至 20.4。失败来自当前 512-D ReLU latent 与 gSDE 尺度耦合，而不是简单的实现崩溃。若继续，需要独立低维/LN 后的 SDE latent 和显式 std 限制。

### Gamma = 0.95

| Gamma | Once | End | Return |
|---:|---:|---:|---:|
| 0.8 | 84.0% | 68.0% | 37.57 |
| 0.95 | 10.0% | 4.0% | 33.51 |

辅助预测仍能下降，但长时域 return/advantage 的方差明显增加，阶段 reward 的局部信用信号被削弱。若继续调 gamma，应从 0.85/0.9 做温和消融，而不是原样复跑 0.95。

### DrAC

只对 Actor RGB 应用共享的轻微 brightness/contrast 变换，以原始策略为 stop-gradient target，Critic 和 B4 辅助目标不变。实现通过模态隔离和数值测试，但完整训练为 0% success，说明当前固定视觉域并不缺少光度不变性，系数 0.1 的一致性约束反而可能压制了精细几何控制特征。

### Beta-PPO

Beta actor 输出 `alpha,beta=softplus(raw)+1`，再从 `[0,1]` 映射到 action bounds；PPO log-prob 和 entropy 包含仿射变换 Jacobian。虽然完全解决动作越界，最终 end 仅 1.3%。这说明动作边界偏差不是当前瓶颈，且策略分布改变需要重新调初始化、entropy 和学习率。

## 9. 统计与验证经验

- 评估 seed 测量一个固定 checkpoint 对随机场景的泛化。
- 训练 seed 改变初始化、探索轨迹、on-policy 数据和 minibatch 顺序，产生独立策略。
- 增加评估 episodes 不能替代增加训练 seeds。
- 本项目用 3 seeds 暴露了 B4-Norm 的高方差；更强结论仍应补至 5–10 seeds，并报告原始散点、bootstrap CI 或 IQM。

## 10. 参考方法

- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- Implementation Matters: https://arxiv.org/abs/2005.12729
- What Matters in On-Policy RL: https://arxiv.org/abs/2006.05990
- Time Limits in RL: https://proceedings.mlr.press/v80/pardo18a.html
- gSDE: https://proceedings.mlr.press/v164/raffin22a.html
- CAPS: https://ai.bu.edu/caps/
- DrAC: https://proceedings.neurips.cc/paper/2021/hash/2b38c2df6a49b97f706ec9148ce48d86-Abstract.html
- Beta Policy: https://proceedings.mlr.press/v70/chou17a.html
- Statistical Precipice: https://arxiv.org/abs/2108.13264
