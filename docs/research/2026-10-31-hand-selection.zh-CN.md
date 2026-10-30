# 收敛选型：LEAP v1 Full作为单一新研究目标

核查日期：2026-10-31；本轮只读资料复核，未安装、运行、训练或采购。协调者根据用户最新优先级形成[ADR 0005](../decisions/0005-research-hand-selection.zh-CN.md)；LEAP 接入任务已准备，尚未派发。

**主推荐：采用 LEAP v1 Full 作为新的统一手部研究目标，先做触控板仿真与决策/训练方法；现有 Allegro 只保留回归参考。** 不建议同时维护四种主训练手。这个选择优先满足“把时间花在决策/训练方法”，并不声称LEAP是G1/H2的现成官方配件。

LEAP v1的优势是有左右手MJCF、独立16关节、公开硬件设计和现成MuJoCo Playground任务，适合改指尖与控制接口；官方组装价约US$2,200（约¥1.46万元，最终Full电机/配件/左右手须确定）。相比移植一套新URDF并重新调驱动和接触模型，它更有利于尽快形成同一研究基线。未来G1适配保留为独立工程阶段；如果用户随后把“现在就必须是G1原厂支持”改成硬条件，首选应切换到Inspire DFX，而不是继续声称LEAP即插即用。

## 四个候选的实际资源与平台支持

| 候选 | 当前已核资源与许可 | G1/H2对应关系 | 本轮判断 |
|---|---|---|---|
| **LEAP v1 Full** | Menagerie有2023 v1构型左右手**简化MJCF，MIT**，来自dexsuite URDF；MuJoCo Playground有LEAP任务，仓库**Apache-2.0**；官方CAD/URDF/API、Isaac训练参考。**官方LEAP_Hand_API是CC BY-NC 4.0，非MIT** | 未核得命名G1/H2整机官方支持；可设计法兰、供电/走线和驱动接入 | **主推荐用于研究**：现成MuJoCo入口最多，独立两指动作空间较宽，成本适中。不是零工程量的精准Full数字孪生 |
| **Inspire RH56DFX** | Unitree XR仓库有左右手Inspire URDF、DFX控制服务、遥操作；官方Isaac Lab有G1-Inspire/H1-2-Inspire任务。XR代码Apache-2.0，但资产及其上游许可仍逐项核。通用Inspire URDF尚未逐参数证明对应当前RH56DFX-2L/2R；**未核到可直接替用的精确DFX MuJoCo/Playground任务** | **G1+DFX官方开发文档与服务支持明确**；FTP也是独立支持选项，不能自动当DFX/E2。H2手名单仍需精确配置确认 | **未来G1实物优先候选**：国内系列采访预算约¥1.8–2万，但不是精确DFX报价。540g轻，移植到MuJoCo与碰撞/耦合标定会增加工作 |
| **BrainCo Revo2** | BrainCo官方左右手URDF/ROS2可视化，**Apache-2.0**；Unitree6驱动串口→DDS服务；XR的BrainCo模型明确V2。**本轮没核到Revo2的MuJoCo Playground现成任务**；Revo3的RevoLab/Revo-MjLab不能拿来替代Revo2 | **Unitree官方明确G1可配Revo2**；不以本体H2与BrainCo末端分别受支持推断H2+Revo2已验收 | 383g，Basic海外单手$3,400–5,400；对当前研究没有比LEAP更直接的MuJoCo优势。作为G1轻量实物备选 |
| **Unitree Dex3-1** | 官方左右手Dex3 URDF、SDK2 G1手控制、G1-Dex3遥操作及Isaac Lab任务；XR代码Apache-2.0。官方unitree_mujoco仓库BSD-3-Clause，但当前`g1_29dof.xml`本轮只见**rubber_hand静态手外形，没有Dex3手指关节**，不能称已拿到精确可动Dex3 MJCF | **G1 EDU+Dex3-1官方产品和软件支持最明确**；H2/R1配件配置有经销商证据，但不是本轮官方组合验收 | 若原生G1工程便利是第一优先可考虑；目前商家配置约$7,680–8,680且单位/套件需最终确认，价格与两指动作自由度均不占当前研究优先级 |

## 两个必须明确的模型/授权边界

**Full不是Lite。** LEAP官方部件页明确：Full使用 **16×XC330-M288-T**，Lite使用 **XL330-M288-T**。我们冻结的Playground README却列XL330，其PD表链接又指XC330。不能声称默认任务已经精准校准Full。Menagerie也自称“simplified”，其转换说明包括惯量修正、执行器设定、简化碰撞和指尖高摩擦。因此这里复用的是**构型、模型文件与任务代码起点**，需要核对Full的力矩/速度/阻尼/惯量，并针对触控板重新设定指腹和滑动摩擦。这是有边界的配置工作，不是重新研发整只灵巧手。

**模型、任务代码、硬件API的许可分开。** Menagerie的LEAP模型MIT；Playground Apache-2.0；官方LEAP_Hand_API `LICENSE` 明写 **CC BY-NC 4.0**，须署名、标注修改且限非商业用途。公开研究项目不能把该API原样复制后整体重新声明MIT/Apache。当前可引用外部依赖并保留条件，或基于合法可用的Robotis协议/SDK单独实现薄驱动；具体采用组件时再核对其许可证。本轮没有写驱动或改变仓库许可。这个区别不妨碍优先使用宽松许可的MJCF与Playground做研究。

## 为什么不直接选G1已支持的六驱动手

不是因为六驱动手“不能捏合”。Inspire/Revo2可利用拇指对指、屈伸和腕部运动改变接触点间距；Dex3也可通过三指构型及腕姿完成部分捏合路径。真正的问题是**选定两指持续贴面时，变距、平移、独立抬指与轻触力是否还能独立调整**。这些应由同一任务的可达性和接触实验判断。

LEAP较多的独立关节减少这种结构约束，便于研究决策、记忆、动作块与闭环训练。将来换到Inspire/Revo2时再通过能力约束、重定向和短期适配处理，不需要为了假定的人形型号提前牺牲本阶段研究灵活性。

## 建议的实施边界

1. Kimi下一步只导入 **LEAP v1构型/Full目标参数**，冻结模型与任务commit；保留Allegro已知场景用于回归。
2. 先验收指尖可达、两指持续接触/变距、独立抬落及稳定滑动，再比较决策/训练方法。允许使用拇指，不把禁用拇指当主目标。
3. 上层任务、记忆和屏幕反馈接口保持与手型号分离；下层通过手能力描述与动作映射适配，避免把手关节索引写进上层任务语义。
4. **不因该研究选择立即购手或承诺G1/H2适配。** 确定未来本体和预算后，在官方G1可接入的DFX/Revo2与LEAP定制适配之间再做交付成本比较。“H2 Plus”应先澄清具体为经销商H2 EDU PLUS配置还是其他版本；官网当前只列H2/H2 EDU。

## 来源与冻结版本

- [LEAP官方Full/Lite部件定义](https://v1.leaphand.com/parts)、[组装购买参考](https://v1.leaphand.com/purchase)。
- [Menagerie LEAP说明与MIT，冻结f3b6013](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/leap_hand)。
- [Playground LEAP README，冻结12ad7f3](https://github.com/google-deepmind/mujoco_playground/blob/12ad7f3842791ced39f27e5ac96d066bbe361653/mujoco_playground/_src/manipulation/leap_hand/README.md)。
- [LEAP API实际许可证](https://github.com/leap-hand/LEAP_Hand_API/blob/main/LICENSE)，本轮直接读取，非商业条款不可忽略。
- [Unitree XR，冻结817fb00](https://github.com/unitreerobotics/xr_teleoperate/tree/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b)：`assets/inspire_hand/`、`assets/brainco_hand/`、`assets/unitree_hand/unitree_dex3_{left,right}.urdf`。
- [Unitree Isaac Lab，冻结e30c25b](https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc)。
- [官方G1+Revo2服务](https://github.com/unitreerobotics/brainco_hand_service)、[BrainCo Revo2模型/Apache-2.0](https://github.com/BrainCoTech/revo2_description)。
- [当前Unitree MuJoCo G1 29DoF XML](https://github.com/unitreerobotics/unitree_mujoco/blob/main/unitree_robots/g1/g1_29dof.xml)：本轮核到rubber_hand，未发现可动Dex3手指，不能由机器人名推断灵巧手版本。

价格细节见[市场价格](2026-10-31-hand-market-prices.zh-CN.md)；人形证据分级见[人形适配报告](2026-10-31-humanoid-integration.zh-CN.md)。未发现精确资产或组合表示本轮未核实，不表示其不存在。
