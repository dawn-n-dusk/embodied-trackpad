# ADR 0011：Sharpa 右手、完整机械臂和动作—结果素材

日期：2026-11-17。状态：用户方向已确定；代码、素材和仿真须分别验收。

## 用户决定

- 用成品 SharpaWave 右手和完整7自由度机械臂；具体采购型号未定。相机包括屏幕相机和腕部相机。
- 第一阶段仍是单指点按、滑动、可靠抬起与屏幕反馈。后续扩展拖拽、多指及独立TrackPoint路线。
- 示范扩充、BC和学生状态纠错打基础，后续明确安排PPO或其他RL；world model作为有对照的研究方案。
- UniDex展示不作为本项目闭环能力标杆；不能因演示视频推定其全系统是否硬编码。
- 完整链路贯通后扩大运行；不把反复零散小测试当成任务成果。
- 两个视频均提取主要手部动作和对应可见结果，裁剪为独立素材。

## 本轮实施合同

两个源为 [Sebtech](https://www.youtube.com/watch?v=lMqXhcpIXSc) 与 [Show Me on Mac](https://www.youtube.com/watch?v=tDa8m3euZU8)。私人素材库保留源媒体hash、来源时间、起止时间与crop坐标；公开仓库仅保存代码、规范和非私有报告。

每段区分真实拍摄、OS教学动画、屏幕录像及口述。手部与屏幕结果如果同一时刻可见，保留原始时间对齐；跨镜头先后展示只能记教学关联，不能伪装同步因果示范。没有结果画面的片段标记unpaired。主要交付包括hand/result裁剪、配对回看片、manifest和离线索引。

这些是BC的来源素材，不自带29维机器人动作、接触力或真实3D手姿。只有后续经过Sharpa/机械臂重定向、执行器驱动、物理接触及GUI结果验证的轨迹，才能作为机器人动作BC候选数据；该阶段另报成功数及失败分母。普通视频关键点是有置信度的估计，不能冒充实际机器人标签。

主场景候选为Isaac Lab/Isaac Sim+PhysX，使用官方Sharpa右手资产及完整7轴臂。臂的临时仿真型号与最终采购型号分开。板面为独立触控板、后方屏幕、右侧偏后的机械臂、随腕运动相机与独立屏幕相机。资产版本、实际mesh payload、法兰变换、关节/执行器顺序均须记录。

保留动作链：关节目标→控制器/执行器→物理接触→设备识别代理→GUI。模拟电容识别尚未实物标定。屏幕相机与GUI framebuffer分开；代理力与完整触觉图像分开；静态原生渲染、真实动态仿真、脚本控制器和学习策略分别记账。

## 本轮不自动启动的事项

本轮搭建和素材提取不等于已经训练模型，不恢复旧训练/自动化，不操作真实手、机械臂或个人桌面。现有Mac VM保留作后续真实macOS验证；大批量训练GUI不是macOS克隆。运行资源以当前有界任务和真实使用回执为准。

## 后续可比较的方法

BC；BC+学生状态纠错；上述基础上带行为约束的PPO；再加入世界模型表征的策略。固定独立任务/布局/设备参数切分。ACT动作块或时间集成不能直接套不匹配的单步PPO概率比。世界模型带来的泛化收益须实测。

参考：[官方资产](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml)、[TacMap](https://github.com/sharpa-robotics/sharpa-tacmap)、[官方RL](https://github.com/sharpa-robotics/sharpa-rl-lab)、[WM-Craftnet](https://github.com/sharpa-robotics/WM-Craftnet)、[Manus接口](https://github.com/sharpa-robotics/sharpa-manus-sdk)、[摄像头/Quest接口](https://github.com/sharpa-robotics/sharpa-vision-retargeting-sdk)。它们是接入与方法参考，不是已完成触控板策略。
