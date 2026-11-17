# SuperMimic 迁移到 Sharpa 触控任务

核查日期：2026-11-18 HKT。已读论文方法部分、官方策略/世界模型学习器、Dex 实现的网络/控制状态，以及 Sharpa 迁移的方法和完整结果表。以下上游成绩均为作者报告，未在本项目复现。

## 可以采用的核心

SuperMimic 使用普通仿真器采集转移，拟合可微动力学模型，再固定世界模型参数，通过短时域想象轨迹的 BPTT 更新策略，并以 critic 补足远期价值。梯度经过神经网络世界模型，原始物理仿真器无需可微。其原任务是人形运动跟踪，尚不能直接证明屏幕反馈触控能力。[论文](https://arxiv.org/abs/2610.28480)、[官方实现](https://github.com/HybridRobotics/supermimic)

官方源码固定于 `7e94b073940b7a19b9a82ff01e191cfb35c4ba91`。可移植重点是 `learners/wm_learner.py` 的多步拟合、`learners/policy_learner.py` 的策略梯度与 critic 分工，以及 replay 的 episode 边界。冻结模型参数时仍须保留对动作和输入状态的梯度；不能用整个 `no_grad` 包住想象过程。[策略学习器](https://github.com/HybridRobotics/supermimic/blob/7e94b073940b7a19b9a82ff01e191cfb35c4ba91/src/supermimic/learners/policy_learner.py)

## 两份灵巧手实现

| 实现 | 已有内容 | 对本项目的价值与边界 |
|---|---|---|
| [CumulusAlpha/Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic) | Wuji Hand 1/2 方块重定向、转笔；MuJoCo Warp/mjlab；发布检查点 | 借鉴失败转移采样、控制器历史状态、接触预测和在线刷新。是不同手型、不同任务，权重不能直接驱动 Sharpa |
| [Hxxxz0/Supermimic-dex](https://github.com/Hxxxz0/Supermimic-dex) | Isaac/ConTrack；XHand 与 Sharpa；有 BPTT/PPO 对照、来源和评测记录 | 更接近我们的机器人与物理平台。可参考完整关节/接触建模、真实与想象奖励核对；仍属于物体操作轨迹跟踪 |

Dex 固定版本 `2becb7380fed86171056f9bf4338b2a6a0c14b01`；Sharpa 迁移仓库固定版本 `c5c9f5ca2923d6c63641f9a3ff453b3b62ebaf6c`。官方代码 Apache-2.0，后者标 BSD-3-Clause 且资源另有许可；前者未确认顶层许可证，因此先参考方法，不直接搬运其源码到公开仓库。[第三方说明](https://github.com/Hxxxz0/Supermimic-dex/blob/c5c9f5ca2923d6c63641f9a3ff453b3b62ebaf6c/THIRD_PARTY_NOTICES.md)

Sharpa 上并非 BPTT 普遍更好：作者报告 CubeMedium 达 99.78%，而 waffleiron 为 0.73%，对应 PPO 为 81.69%。其表共有 26 个任务/方法组合，21 个完成最终评测，单训练 seed；未评测不能算失败率为零。这是保留 PPO 对照、检查接触模型失配的直接理由。[完整结果](https://github.com/Hxxxz0/Supermimic-dex/blob/c5c9f5ca2923d6c63641f9a3ff453b3b62ebaf6c/docs/SHARPA_RESULTS.md)

## 本项目的迁移决定

继续使用现有 Sharpa＋7 轴臂和 Isaac 接触环境。世界模型路线调整为 **真实转移 → 多步世界模型拟合 → 想象中的 BPTT actor/critic → 回到真实仿真器刷新数据和评测**。MPC 候选保留，首轮重点放在 BPTT。

我们仍要求策略看屏幕和腕部 RGB。上游的物体真值、参考轨迹、物理随机化参数不能直接塞进视觉 actor。世界模型可在训练时使用额外监督，但部署 actor 的输入仍严格受限；若单独运行状态策略，只作为标明特权信息的诊断基线。

触控板有相对位移、落指基准、点按计时、移动阈值和释放等记忆，必须用时间状态或显式设备记忆建模。只预测下一帧画面或关节位置不足以解释点击。接触切换和点击采用的可微近似必须与真实设备适配器分别验证，最终成功由实际接触及 GUI 结果决定。[Sharpa 迁移方法的真实/想象语义处理](https://github.com/Hxxxz0/Supermimic-dex/blob/c5c9f5ca2923d6c63641f9a3ff453b3b62ebaf6c/docs/METHOD.md)

本轮采用相同初始数据和独立测试，分别记录 BC、纠错、PPO 与 BPTT 策略的实际交互量、模型更新和完整成功率。先检验时间展开、动作梯度、接触预测与真实闭环，再提高并行规模；论文的人形加速比不作为本项目的预计加速比。

## 迁移实现的数学验收

有限时域 actor 目标采用

$$J=\sum_{t=0}^{H-1}\gamma^t\left(\prod_{k=0}^{t-1}c_k\right)r_t+\gamma^H\left(\prod_{k=0}^{H-1}c_k\right)V(s_H).$$

这里的空乘积为 1。末端价值包含全部 H 个 continuation；固定 critic 参数时仍保留价值对末端状态的梯度。critic 的递归目标为

$$G_t=r_t+\gamma c_t\big[(1-\lambda)V(s_{t+1})+\lambda G_{t+1}\big],\qquad G_H=V(s_H).$$

用于 critic 拟合的目标与输入状态脱离 actor 图。三个可手算检查为：一步终止后的价值贡献应为 0；奖励 `[1,2]`、折扣与 lambda 均为 1、末端价值为 0 时，目标应为 `[3,2]`；只有末端价值依赖动作时，actor 仍应获得非零梯度。这些是软件正确性检查，不是机器人训练或任务成功。
