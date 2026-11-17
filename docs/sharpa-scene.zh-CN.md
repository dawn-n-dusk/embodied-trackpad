# SharpaWave 右手与七轴机械臂仿真场景

2026-11-17：原生场景与脚本运动验证通过。最终运行 run-20261117T141130Z 测得完整 29 自由度，推进 1680 个物理步、14 秒；三路相机画面均由协调者实际查看，两路录像各 210 帧、15 fps、14 秒。验收范围是场景、运动与记录，不包括接触任务成功或学习策略。

## 模型与布局

- 官方 SharpaWave 右手 with_flange，版本 d323b371d02f4fc04a0873c19c5234cae6d803dd，22 个驱动通道。
- NVIDIA 官方 Franka Panda USD 临时代表完整七轴臂，不代表采购决定。旧 Panda 手爪已停用，实际关节数为 7+22=29。
- 桌面高 0.72 m、尺寸 1.20×0.80 m；触控板中心位于 (0,0)，尺寸采用 160×114.9×10.9 mm 设计假设，未实物标定。
- 臂基座位于右后方 (0.42,0.30)m。屏幕位于左后方 (-0.38,0.34,1.15)m，屏幕相机同步左移，避开机械臂遮挡。
- 坐标为 +X 向右、+Y 远离操作者、+Z 向上。Sharpa 模型的手指沿局部 Z 延伸、沿局部 Y 排列，掌面法向是局部 X。最终掌面朝下、手指向前略下倾；初始食指参考点目标为板面上方 35 mm。
- 总览、屏幕 RGB、腕部 RGB 三个视角。腕镜头从拇指/食指侧观察手指和触控板。此配置只验收单环境默认布局，批量和安装位置泛化尚未验收。

模型来源：[Sharpa 官方资产](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/d323b371d02f4fc04a0873c19c5234cae6d803dd)、[NVIDIA Panda USD](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd)。资产及依赖单独获取，不随本仓库重新分发。

## 实测结果与边界

运行使用现有 Isaac Sim（包版本 6.0.1.0，启动日志为 6.0.1-rc.7）、Python 3.12.3、NumPy 2.3.1、SciPy 1.17.0、Pillow 12.2.0 和系统 ffmpeg。本轮物理由 CPU PhysX 推进，单张 RTX GPU 负责渲染；没有安装或接入 Isaac Lab 批量训练。

- 841 行实测关节状态，均包含 29 维有限 q/qd，手臂与食指实际运动；不是每步设置关节状态的回放。
- 初始化允许一次设置合法关节状态；后续只设置驱动目标。初始指尖与 IK 预测差 0.117 mm，最终差约 2.212 mm；单次悬空运动不代表精准点击能力。
- 固定关节两侧实测最大位置差约 9.50e-8 m。渲染前后记录的物理步增量为 0；初始化 2 步与正式 1680 步分开记账。
- 食指参考点最小板面高度约 27.39 mm；接触代理事件为 0，点击为 0。没有验收点按、滑动、抬起恢复或其他手指的真实接触行为。
- 最大瞬时关节目标误差约 0.10683 rad，包含目标切换时的跟踪过程，不等同于稳态误差。
- 35 项配置/设备逻辑检查通过，180° 姿态歧义与新初始姿态另经 CPU 数值核查。实际渲染、物理状态和视频时序独立于这些软件检查验收。

屏幕展示几何目标与光标，两者位置对应 GUI 状态；它不是 macOS 桌面，尚未连接 Mac/VM。没有策略优化器或通用 Gym reset/step 包装。当前脚本用于场景验收，不是 BC 或 RL 成果。

## 接口与复现

源码模块为 embodied_trackpad.sharpa_sim 下的 config、device、kinematics、isaac_scene 和 isaac_runner。动作合同是完整 29 维关节位置目标；实际通道顺序随每次运行保存。记录包含命令时刻、实际仿真时刻、物理步、本体状态、相机帧与单独的 GUI 教师信息。

在已有兼容 Isaac Sim 环境中准备完整 Sharpa 目录及 Panda USD 的 Props/configuration 依赖。场景引用本机资产绝对路径，换机器须重新生成。入口示例：

```bash
mkdir -p "$RUN_DIR"
PYTHONPATH=src "$ISAAC_PYTHON" -m embodied_trackpad.sharpa_sim.isaac_scene \
  --hand-usd "$HAND_USD" --franka-usd "$PANDA_USD" \
  --num-envs 1 --out "$RUN_DIR/scene.usda"
PYTHONPATH=src "$ISAAC_PYTHON" -m embodied_trackpad.sharpa_sim.isaac_runner \
  --stage "$RUN_DIR/scene.usda" --output "$RUN_DIR/episode" --episode-seconds 14
```

当前 runner 的 RTX/physics device 索引固定为 0；本轮核验的是物理 GPU0。其他卡的 CUDA/Vulkan 映射需要显式修改与核验，不能只改 CUDA_VISIBLE_DEVICES 后把参数中的 UUID 当成实际分配证据。共享机器运行前核对资源和进程归属。个人 SSH 与调度工具不发布。

设备适配器提供接触事件到相对光标与点按的状态机；物理接触到电容识别仍是未标定代理。默认平板以外的倾斜/旋转与多环境组合尚未端到端验证。教师真值只用于日志、奖励和评测；未来视觉策略的输入契约须排除它，Python 模块本身不构成安全隔离。

下一步是视频姿态提取、完整臂手重定向、物理接触及 GUI 结果验收，再形成机器人 BC 数据与学生状态纠错。原视频参考与当前悬空诊断均不计入机器人动作 BC 成功轨迹。见[素材说明](video-materials.zh-CN.md)及[视频到 BC 规范](video-to-bc.zh-CN.md)。
