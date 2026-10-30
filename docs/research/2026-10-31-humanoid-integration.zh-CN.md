# 人形机器人集成与扩展性核对

日期：2026-10-31 香港；本轮重点复核官方集成资料，沿用同日已存档型号资料。只读调研，没有安装、控制、训练、询价或采购。这里“支持”区分四种事实：**指定组合的官方软件/产品支持、厂商面向人形的设计/通用兼容声明、经销商可选配置、只有定制集成可行性**；均不等于我们已实装验收。

**最省集成工作的是已有指定人形平台支持的手；最方便跨平台研究的则是开放模型、独立关节和可修改结构。这两类优势不完全重合。** 若未来确定G1，可优先看Dex3-1、Inspire DFX/FTP、Revo2；若希望保留跨人形平台的自由度，Tesollo S、LEAP v1、Allegro、Revo3更值得比较，但需要手腕适配和驱动集成。

## 指定平台证据

| 手版本 | 已核的具体人形组合 | 证据级别及未确定项 |
|---|---|---|
| **Inspire RH56DFX / FTP** | **Unitree G1 + DFX；G1 + FTP**。Unitree遥操作README分别给G1开发文档与 `inspire_dfx`/`inspire_ftp` 选项。服务说明还给G1和H1的DFX控制入口 | **官方指定组合的软件支持**。官方Isaac Lab还有G1-29+Inspire、H1-2+Inspire任务，但模型名未把DFX/FTP当前SKU锁定。FTP旧产品URL现在指向E2，不能把旧FTP证据换成E2。 |
| **BrainCo Revo2** | **Unitree G1 + Revo2** | **最明确的一条官方组合**：Unitree `brainco_hand_service` README直接写“The G1 can be equipped … Revo2 Dexterous Hand”，6驱动、USB串口转DDS；并链接BrainCo自己的G1适配仓库。不能继承到Revo3。 |
| **BrainCo Revo3 U21/U21T/U21V** | RobotLAB列 **H2 EDU + U21/U21T/U21V + RGB**、**H2 EDU PLUS + 对应手 + SHF3L相机** | **明确的经销商配置**。本轮Unitree H2官网只称H2 EDU可选多型号手，未点名Revo3；BrainCo官方有ROS2、双臂/臂手研究代码，但本轮未核到命名整个人形+Revo3版本的官方装机证明。经销商U21V与BrainCo官方U21VT命名也需对齐。 |
| **Unitree Dex3-1** | **G1 EDU + Dex3-1**；G1-29 + Dex3官方遥操作视频标题及仿真任务 | **官方产品+软件支持**。RobotLAB还列G1-D/H2/R1适配配置，但不能据此写这些组合都已有同等官方实机验证。触觉/非触觉版本另分。 |
| **Unitree Dex5-1 / Dex5-1P** | RobotLAB列 **H2 + Dex5-1 / Dex5-1P + RGB** | **经销商指定配置**。官网手产品页明确两版本参数，但本轮没找到命名G1或H2的具体安装/驱动验收材料。应纠正旧报告“如果用G1则重点选Dex5-1”的过强联想：可以作为候选，不能称G1官方现成支持。 |
| **Unitree Dex5-S / Pro** | 官网手参数完整；未核得指定G1/H2整机组合 | **厂商手产品已存在，具体人形兼容未落实**。同属Unitree或菜单并列不是装配证明；它不是Dex5-1/P，也不能继承Dex3遥操作资产。 |
| **Tesollo DG-5F-S 20DoF** | 官网明确面向人形整合；未核得命名量产人形+S型号装机 | **人形设计定位+通用ROS2/模型支持**。有15DoF和20DoF资源，应锁20DoF。 |
| **Tesollo DG-5F-M** | 官网有短腕选项和面向人形用途；未核得命名人形+M精确版本 | **通用安装/设计**，非即插即用。重约1.763kg，腕负担明显高于S。 |
| **Allegro V4 / V5 Plus / V6 F** | 本轮未核得命名人形+精确代际组合 | **研究平台和通用手腕集成**。长期机械臂/研究采用并不自动等于人形原厂支持。代际控制总线、模型不同。 |
| **LEAP v1 / v2 / Advanced** | 官方支持机械臂安装/自定义底座；未核得指定人形组合 | **开放硬件改装路线**。可以做转接，但机械设计、电源/线束和动力学标定由集成者承担。v1与v2资产不同。 |
| **Linker O6** | **Linker Humanoid V2**官网称“适配所有型号灵巧手”；RobotLAB列 **R1 + O6（触觉/非触觉）+RGB** | 前者为**同厂全系列兼容声明**（涵盖O6，但不是逐SKU实机演示）；后者是**经销商配置**。不能说O6已在任意人形上即插即用。Linker Humanoid V2是可选轮式底盘的升降双臂平台，不宜写成双足人形证据。 |

官方遥操作README现在把H2、R1、G1/H1等本体与Dex3/Inspire/BrainCo等末端列为各自受支持选项。**两个名单的笛卡尔积不是经过验证的全部组合。** 例如 `--arm H2` 与 `--ee brainco` 都存在，仍不足以证明H2+Revo3：已核的BrainCo控制器是6驱动Revo2。

Shadow Classic/E3M5仍可作为专用研究集成候选，但先前核得约4.3kg口径含驱动/前臂总成。它不是轻型人形腕口的通用替换件，本项目暂不优先采购；不能把它与只报裸手重量的型号直接比较。

## H2和新手：目前能说到哪一步

[H2官网](https://www.unitree.com/H2) 当前表格明确：基础H2的Dexterous Hand栏为“/”，H2 EDU为 **“Dexterous Hand (Multiple Model Options)”**；EDU支持二次开发。额定臂载约7kg、峰值约15kg只是臂级参数，不代表任意手的腕部动态负载、线缆、电源和保修兼容。

[RobotLAB目录](https://www.robotlab.com/store/unitree-humanoid-hands/) 提供了比官网细的H2+Dex5-1/P和H2 EDU(+)+Revo3组合；可以把它当成**有商业集成渠道的候选**，下一步要求对方列出法兰图、适配线束、电源转换、驱动版本与实际交付型号。当前证据不能把它升级为“官方开源控制栈已经完整支持”。

## 扩展性：哪些条件真正影响跨平台改装

| 手版本 | 已核重量、电气与控制 | 公开软件/模型及实际含义 |
|---|---|---|
| DFX | 540g无腕；DFXW 650g带腕；24V±10%、最大2A；RS485 | Unitree桥接/DDS、URDF/任务已有；6主动/12关节，四指缺独立侧摆。轻、成熟对接有利人形，动作空间仍受限。FTP本轮不拿DFX重量代填。 |
| Revo2 | 383g不含腕；RS485/CAN FD/EtherCAT按交付版本核对 | 官方URDF/SDK、G1串口→DDS服务明确；6主动/11运动DoF。低重量和既有G1支持有价值，但相邻指变距能力有限。 |
| Revo3 | 官方当前手册重量未列；**16–85V、最大10A@24V**；RS485/CAN FD/EtherCAT、控制500Hz | 21主动关节，官方SDK、ROS2描述及RevoLab/Revo-MjLab；研究扩展性较强。要核完整手质量/重心、CPU架构、触觉SKU；旧v0缓存不兼容v1。经销商12–80V不能覆盖官方当前手册。 |
| Dex3-1 | 710g；USB，公开G1控制例程 | 最清晰的Unitree原生三指路线；食中指无独立侧摆，触控板变距受限。 |
| Dex5-1/P | 1.1kg；USB2.0；官网列58V电流工况 | 16主动+4耦合；P版94触觉。58V测试工况不能替代允许供电范围；24V腕口是否可用尚未核验，USB协议、准确SDK/模型也须匹配。 |
| Dex5-S/Pro | **620g；15–65V、XT30、峰值200W**；Ethernet/USB控制1000Hz，RS485为50Hz | 22DoF；官网同时写1000BASE-T1配置与T1s Ethernet，术语对应仍需厂商澄清，不等同随便一根普通RJ45线即可连接。低重量/宽电压有利跨平台，但准确驱动、模型和安装图本轮未验。 |
| Tesollo S/M | S880g、500Hz；M1763g、250Hz；均24V、最大10A；S Ethernet TCP/IP/Modbus TCP，M另列Modbus RTU | 官方左右手URDF/MJCF/USD、ROS2、ros2_control/MoveIt资料；S独立20关节是较好的通用候选。官网SDK列Linux x86_64；Jetson ARM是否源码可编译/二进制可用须验证。 |
| Allegro V4/Plus/V6 F | 1080/1024/1150g；V4 CAN333Hz，Plus CAN500Hz；V6 F RS485/Ethernet Modbus；24V约100–120W | 研究生态成熟但版本分裂；项目Menagerie是V3，不能直接代表V4/Plus/V6。四指独立关节有助精细操作，V6新接口需重做驱动匹配。 |
| LEAP v1 | 官方本轮未核到统一重量；Dynamixel串行总线、U2D2/供电组件，按Full/Lite电机配置 | 开放CAD/URDF/API、Menagerie/MuJoCo Playground，改指腹/底座最自由；外形较粗、线束和舵机维护由团队负责。v2/Advanced耦合/腱驱动需另标定。 |
| O6 | **360g、24–48V；CAN/CAN FD/RS485/ECAT；6主动+5被动**，触觉选配 | 官方页面列Windows上位机、Ubuntu示例；本轮未核到准确O6可用URDF/MJCF和训练例程。厂家接口列表仍须与订单具体控制板对应。四指弯曲+拇指侧摆，不能当全独立侧摆的五指手。 |

跨平台选择需先核**腕法兰与走线、重心和惯量、电压/峰值功率、CPU/OS驱动、位置/速度/电流或力控制、带传感器的准确模型**。ROS2、USB或EtherCAT相同只减少一部分工程量，不能免去这些核对。

## 针对当前项目的判断

- 如果未来人形尚未定，**LEAP v1适合作低成本可改装研究，Tesollo DG-5F-S 20DoF适合作商品化独立关节候选**；Allegro保留成熟对照，Revo3保留高预算候选。这些是扩展潜力排序，不是指定人形安装完成度排序。
- 如果确定Unitree G1，**Dex3、DFX/FTP、Revo2的官方对接证据最强**。其手指动作限制要和触控板任务一起考量，不能单为容易安装牺牲目标动作。
- 如果确定H2，**Dex5-1/P、Revo3已有经销商配置线索，但应先获得版本明确的官方/供应商交付接口清单**。Dex5-S值得关注，暂不能称其现有G1/H2软件资产已成熟。
- O6是轻量低价路线，且有同厂人形兼容声明；但针对本项目最需要的独立两指变距，不能用“能安装”代替“能完成”。

## 对后续软件架构的影响

保留“屏幕观察、任务规划与记忆”作为共同上层；每种手分别配置执行器布局、关节/驱动映射、传感器、腕部坐标变换和控制适配器。更换手后，上层任务接口可以复用，低层运动策略通常仍需重新训练或微调，不能因为ROS2接口相同就假定旧权重直接有效。

机械接入还需要把手、转接件和线束的质量/重心/惯量加入机器人模型，并检查腕部持续扭矩、瞬态电流和线缆活动范围。若厂商只提供x86_64二进制，要验证Jetson/其他ARM计算平台是否能从源码构建，或采用独立控制计算机经网络连接；当前资料没有证明ARM受支持，也没有证明ARM一定不能用。

这是一份选型与接口设计建议。本轮没有更换现有Allegro仿真基线，没有安装任何新手或修改真实机器人。

## 本轮官方来源

1. [Unitree BrainCo Hand Service](https://github.com/unitreerobotics/brainco_hand_service)：明确G1+Revo2及6驱动串口/DDS。
2. [Unitree XR teleoperation](https://github.com/unitreerobotics/xr_teleoperate)：G1+DFX/FTP、G1+Dex3、H2/R1本体支持；不是任意组合兼容矩阵。
3. [Unitree Isaac Lab](https://github.com/unitreerobotics/unitree_sim_isaaclab)：G1-Dex3、G1-Inspire、H1-2-Inspire任务。
4. [G1官方](https://www.unitree.com/g1)、[H2官方](https://www.unitree.com/H2)、[Dex5-1/P](https://www.unitree.com/Dex5-1)、[Dex5-S/Pro](https://www.unitree.com/Dex5-S)。
5. [Revo2官方手册](https://www.brainco-hz.com/docs/revolimb-hand/revo2/parameters.html)、[Revo3官方手册](https://www.brainco-hz.com/docs/revolimb-hand/revo3/parameters.html)、[Revo3 ROS2](https://github.com/BrainCoTech/brainco_revo3_ros2)。
6. [Tesollo S](https://www.tesollo.com/products/grippers/dg-5f-s)、[M](https://www.tesollo.com/products/grippers/dg-5f-m)、[官方模型](https://github.com/tesollodelto/tesollo_model)。
7. [Allegro V4](https://www.allegrohand.com/sub/product/p.php?idx=1)、[V5 Plus](https://www.allegrohand.com/sub/product/p.php?idx=4)、[V6 F](https://www.allegrohand.com/sub/product/p.php?idx=25)。
8. [LEAP v1](https://v1.leaphand.com/)、[Inspire DFX](https://www.inspire-robots.com/dexterous%20hands/rh56dfx-series/)。
9. [O6官方](https://www.linkerbot.cn/details/LinkerHandO6)、[Linker Humanoid V2官方](https://www.linkerbot.cn/details/LinkerHumanoid)。
10. [RobotLAB配件目录](https://www.robotlab.com/store/unitree-humanoid-hands/)：**经销商证据**，与官方证据分级保存。

[配套证据](2026-10-31-humanoid-integration-evidence.json)记录原文、证据类别和源文件哈希；原始网页留在私有审计目录。旧型号参数沿用[同日候选研究](2026-10-31-hand-candidates.zh-CN.md)官方存档。Unitree开发文档直连本轮567错误，DFX服务main分支路径404；不据这些失败推断产品不支持。未全面搜索全部论文和厂商演示，未发现命名组合表示本轮未确认，不表示该组合不存在。
