# 灵巧手候选筛选：触控板操作与未来人形集成

> 人形接入更新：[指定平台与跨平台扩展核对](2026-10-31-humanoid-integration.zh-CN.md)澄清G1与H2、新旧手代际的支持边界；Dex5同属宇树不构成G1原生支持证明。

> 价格更新：本页保留首轮研究口径。新增国内媒体采访、商家挂牌、历史与二手检索结果见[市场价格补充](2026-10-31-hand-market-prices.zh-CN.md)；其中已补齐 Revo、Allegro、Tesollo S 等公开预算，不再仅以官方是否列价判断价格未知。

核验日期：2026-10-31（香港时间）。范围：7 个厂商/开源系列，区分具体硬件代际、仿真资产、公开价格和采用证据。仅做资料研究；没有采购、联系厂商、运行仿真或操作机器人。

**建议先仿真比较 LEAP v1 与 Tesollo DG-5F-S 20-DoF，同时保留现有 Allegro V3 作为对照。未来实物优先询价 LEAP v1、DG-5F-S、BrainCo Revo3 U21F/U21T；若确定使用 Unitree G1，则把 Dex5-1/P 加入重点询价。** 目前不能由一次不顺畅的演示判定 Allegro 硬件不好用：工程中的安装姿态、关节坐标、接触模型和控制器，也会制造“手很笨”的观感。应先统一这些条件再比较。

“普遍好用”没有可靠的跨品牌排行榜。本报告把它拆成研究采用、软件可获得性、触控板动作能力、维护负担与人形集成五项；没有把厂商宣传的最大握力当成触控板性能。**所有候选的真实 Mac 电容触控板兼容性均未被本次研究验证。**

## 1. 哪些候选值得看

主动自由度指可独立命令的电机/关节数；运动自由度包含机械耦合或欠驱动关节，二者不能混用。重量按厂家不同口径保留，不能直接将含前臂总成与裸手比较。

| 候选与具体硬件版本 | 构型、重量 | 对本项目的优势 | 主要不足与边界 | 人形机器人集成证据 |
|---|---|---|---|---|
| **LEAP v1 Full** | 4 指、16 主动关节；本次官方网页未核得统一整机重量 | 开源硬件/控制/训练资料丰富；MCP 侧摆在手指伸直或弯曲时都能使用；便于改指尖，价格最透明。适合作为第一种新基线 | 外置舵机结构较粗；齿隙、低速摩擦和指尖尺寸需实测；Full 与 Lite 电机不同；不是现成的人形手腕即插即用产品 | 官方提供机械臂安装/定制底座；本次未确认具体人形平台装机。属**需设计转接的候选** [L1][L2] |
| **LEAP v2 Low-cost / v2 Advanced** | Low-cost：4 指、8 电机/16 运动 DoF；Advanced：17 电机/21 运动 DoF，含 2 个主动掌部关节 | 柔性指段可降低碰撞损伤；可自行制造维护；两种版本均强调 MCP 侧摆 | 腱驱动/耦合指段带来迟滞、柔性和标定问题；不能继承 v1 的模型或训练成熟度。Advanced 制造和装配复杂，主页低价与零件表不一致 | 官方称可装机械臂；未核得具体人形装机。适合低成本/柔性结构研究，**暂不优先替换主基线** [L3–L6] |
| **Allegro V4 / V5 Plus / V6 F** | V4：4 指16 主动、1.08 kg；V5 Plus：4 指16 主动、1.024 kg；V6 F：5 指20 主动、1.150 kg | 旧四指系列有长期研究基础；侧摆与屈伸可独立控制；V5 Plus 加硅胶/压力指尖，V6 F 提供更完整五指与接触传感 | 齿轮传动的低力控制要调；现有项目模型是 **V3**，不能拿来代表 V4/Plus/V6 F 的动力学。**V5 非 Plus 是9 DoF**；V6 F 是较新产品，不能继承旧代全部证据 | 可通过手腕转接集成；本次未建立具体型号与人形平台的装机证据。V4/Plus 更偏研究平台 [A1–A4][A6] |
| **Inspire 因时 RH56DFX-2L/2R**；E2另列 | DFX：5 指、6 主动/12 关节、540 g（不带腕）；带腕 DFXW 650 g、6+2 DoF。E2：790±10 g、6/12 | 轻，通信/控制接口已有应用，Unitree 有明确 G1 对接。对简单点按、滚动和抓取，集成负担较低 | 食/中指各自卷曲，缺少独立侧摆；用相邻两指保持贴面并连续改变间距，约束较大。可改用拇指与另一指，不能说“完全不能缩放”。DFX 力分辨率0.5 N是测量规格，非轻触控制保证 | **Unitree 官方 G1 文档及遥操作明确支持 DFX/FTP**。不能把该证据直接套给新 E2；旧 `rh56dftp-series` 产品网址如今展示 E2 [I1][I2][U3] |
| **BrainCo Revo2 / Revo3 U21系列** | Revo2：5 指、6 主动/11 运动 DoF、383 g不含腕；Revo3：5 指、21全主动、216×108×50 mm；本次官方手册未列重量 | Revo2 极轻且有官方模型；Revo3 每个非拇指4独立DoF、拇指5DoF，支持侧摆、可反驱、阻抗/电流模式，500 Hz；对触控板的构型比 Revo2 更合适 | Revo2相邻指间距调节受限；Revo3较新，重量、价格、长期低力重复接触性能需核对。U21基础版无触觉，F/T/VT传感器不同。0.01 N触觉分辨率不等于0.01 N闭环控力 | Unitree遥操作资产明确为 **BrainCo V2**；Revo3官方有臂手/双臂科研集成与软件，但本次未确认完整人形装机。不能用V2证据替Revo3背书 [B1–B5][U3] |
| **Unitree Dex3-1 / Dex5-1(P)** | Dex3：3指7主动、710 g，175×88×77 mm；Dex5：5指16主动+4耦合、1.1 kg，217.3×127.5×72.1 mm | Dex3在G1生态中开源对接明确、USB 1kHz；Dex5四指有±22°侧摆，比Dex3更适合相邻指变间距。Dex5-1P带94个触觉单元 | Dex3食/中指各2个同轴屈伸关节，无侧摆；两指相对横向运动受限。Dex5较新，本次没有确认版本匹配的公开训练/模型。**94个触觉单元属于P版，不是基础版** | **Dex3官方明确用于G1**，且有G1仿真/遥操作任务；Dex5需进一步确认具体G1版本、腕部转接及电源兼容 [U1–U4] |
| **Tesollo DG-5F-S 20-DoF / DG-5F-M** | S：5指20独立关节、880 g、500 Hz；M：5指20独立关节、1.763 kg、250 Hz | 每指4个独立关节，适合把侧摆、贴面、抬指分开；官方提供URDF/MJCF/USD；S减重明显且明确为人形集成设计；M有模块化指尖传感器/短腕版 | S较新，公开模型参数仍注明可能更新；单看模型齐全不能确认sim-to-real精度。售价未公开。M较重、对本项目最大握力富余，腕部负担大 | **厂家明确人形设计**：S为人形优化，M有短腕版；本次未确认S已装某一命名人形平台 [T1–T4] |
| **Shadow Dexterous Hand（Classic；模型E3M5）** | 5指、20主动/24关节，含2腕DoF；整套约4.3 kg | 长期灵巧操作研究基础最强的一档；独立侧摆、腱传动、丰富传感，ROS/EtherCAT 1kHz。可作高能力仿真参考 | 前臂/驱动总成大、重，维护和集成复杂；对“先实现触控板操作”投入偏大；无当前公开价格，不能使用网上流传的十几年前报价 | 可研究级定制集成，但不能当成轻型人形手腕通用件。4.3kg口径含其驱动/前臂结构，采购时需确认总成边界 [S1][S2] |

这些候选中，**Allegro旧四指、LEAP v1、Shadow** 有最清晰的长期研究和可复用模型证据。**Inspire DFX、Unitree Dex3、BrainCo V2** 的优势是现成的人形软件集成，但手指运动受耦合或自由度约束。**Tesollo S、BrainCo Revo3、Unitree Dex5、Allegro V6 F** 更值得关注其新构型，但不能把“新参数更好”写成“已经普遍验证更好用”。

## 2. 价格：可核验的数字与不能确定的部分

以下均为2026-10-31读取的公开页面；页面内历史BOM日期另列。美元保留原币，不使用未核验汇率换算。**包含官方成本/组装价与另行标明的零售参考；本次未取得正式报价或结账价格。** 除特别标注，税、运费、进口费用、保修、备件、手腕法兰和电源装箱范围均需正式报价确认。

| 型号 | 可核验价格 | 报价口径/原文与时间 |
|---|---|---|
| LEAP v1 Full 自制 | 主页约 **US$2,000**；零件页历史合计 **US$1,771.74** | 原文 “assembled … at a cost of 2000 USD”；BOM标注 **Aug 2023**，含16电机、U2D2、供电Hub、电源与线材等，工具/打印机另列；不是2026成品现价 [L1][L2] |
| LEAP v1 组装手 | **around US$2,200** | 官方购买页原文 **“Around ~$2200 for assembled hands”**，通过邮件选择左/右手和已组装/零件包。页面未给可结账SKU或完整装箱清单；单/双手计价、税运需最终确认 [L7] |
| LEAP v1 Lite | 历史BOM **US$714.30**（页面概称~$700） | **Aug 2023**，使用较弱/较不耐久XL330电机；不能当成Full性能的等价廉价版 [L2] |
| LEAP v2 Low-cost 自制 | 主页$200；零件页更新称 **around $300**；列出的单手采购项 **$221.95 + $251.53** | 原文 **“parts cost around $300 now due to tariffs”**；表格 **“Total Cost (1 Hand): As of August 2025”**。后项含可制造多只手的整包耗材，首次采购和摊销成本不同。$200是设计成本宣传，不能当成成品售价 [L3][L4] |
| LEAP v2 Low-cost 组装手 | 官方零件页写 **~US$500**，3D打印件包 **US$100** | 原文 **“limited supply of 3D printed parts kits ($100) or fully assembled hands in stock (~$500)”**；需联系确认库存与装箱，未验证现货或最终支付价 [L4] |
| LEAP v2 Advanced 自制 | 零件明细：塑料版 **US$4,079.88**；金属件版 **US$5,079.88** | BOM标注 **Dec 2024**。主页“$3000”、零件页顶“metal palm … around $4000”与明细冲突；预算应保留明细并重新询价，不能承诺$3000能做成整机。工具/打印机另列 [L5][L6] |
| Allegro V4 / V5 Plus / V6 F | **需询价** | 本次官方产品/销售页面未见当前公开价格 [A1–A3] |
| Inspire DFX / E2 | DFX左手零售 **€4,150未税 / €4,980含税**；E2需询价 | Génération Robots具体左手SKU；该品牌授权关系及附件未独立确认，属于零售参考，详见[价格补充](2026-10-31-hand-prices.zh-CN.md)。官方页本身未公开金额 [I1][I2] |
| BrainCo Revo2 / Revo3各触觉版 | **需询价** | 官方手册有配置区分和装箱说明，但没有公开价格 [B1][B2] |
| Unitree Dex3-1 / Dex5-1(P) | **需询价** | 官方页面无可核验售价；不同手型、左右手、触觉版和G1套餐不可混报价 [U1][U2] |
| Tesollo DG-5F-S / M | **需询价** | 官方S页只写约为M价格的**60%**，M基准金额未公开，不能据此编造售价 [T1][T2] |
| Shadow Classic | **需询价** | 本次官方产品页仅销售咨询；不沿用历史二手/媒体价格 [S1] |

另查到 **Unitree Dex5-S官方标价US$6,350，Pro变体数据US$13,500**，页面称12月开始发货、单只/一对及装箱未明确；**不能用于Dex5-1/P**。MYBOTSHOP的旧型号 **Tesollo DG-5F单手套装标价€18,886.95（含19%德国VAT、运费另计）**，**不能用于DG-5F-S**，也不按厂家“60%”相对描述换算S价格。准确SKU、供货状态、税费和核验链接见[价格补充报告](2026-10-31-hand-prices.zh-CN.md)及[机器可读证据](2026-10-31-hand-prices-evidence.json)。

LEAP的廉价是开放制造路线的优势，但需要计入打印设备、装配工时、校准、损坏备件及更换电机的时间。对本项目而言，公开组装价与可改指尖比最大握力更有价值。

## 3. 我们实际能拿到哪些仿真资产

这里的“存在”仅表示读取到了公开文件/说明；**没有完成加载、稳定性、碰撞精度或GPU支持验收**。所有使用都应冻结commit，再记录控制器、碰撞几何、单位及指尖改造。

| 硬件家族 | 本次核验的实际资产版本 | 格式/许可 | 证据成熟度与限制 |
|---|---|---|---|
| Allegro | 项目现有Menagerie为 **Allegro V3**，不是V4；冻结 `f3b6013…` | 左右手MJCF；BSD-2-Clause | README明确“simplified … V3”，加入位置执行器且缩放过质量；V4/V5/V6需另取准确模型。HORA公开V4例程是另一条资产链 [A4][A5] |
| LEAP | Menagerie对应**2023 LEAP v1构型**，不是v2/Advanced | 左右手MJCF；MIT；来源为dexsuite URDF；同一冻结commit | 自带简化碰撞和指尖高摩擦设定，需针对滑动重新标定。MuJoCo Playground有LEAP转动/重定向任务；其说明列XL330电机，不能直接当Full电机标定 [L8][L9] |
| Shadow | **E3M5**，不是所有Shadow新产品 | 左右手MJCF；Apache-2.0；同一冻结commit | 官网Classic参数与E3M5模型仍需逐项对齐；强研究基础不代表我们的触控板场景已验证 [S3] |
| Tesollo | 官方 `dg5f` 与 **`dg5fs`**；S明确分20DoF与15DoF | URDF/MJCF/USD；BSD-3-Clause；仓库树 `84be60a…` | S有左右手/带或不带底座文件；README提示参数会更新。必须选20DoF，避免误用15DoF；不是只有M模型 [T3][T4] |
| Unitree / Inspire | `xr_teleoperate` 有 **Dex3** 与Inspire 6驱动/12运动关节URDF；官方Isaac Lab含G1-Dex3、G1-Inspire、H1-2-Inspire任务 | URDF/官方Isaac配置；遥操作仓库Apache-2.0，附第三方许可说明；树 `817fb00…` | Inspire资产名未绑定当前E2 SKU；Dex5未在本次核验的任务与资产路径中确认。不将“有厂家产品页”写成“已有对应训练模型” [U3][U4] |
| BrainCo | 官方Revo2 URDF；Unitree模型名为 **`brainco-righthand-V2`**。Revo3有独立ROS2描述/SDK，RevoLab、Revo-MjLab、RevoSim | Revo2/Revo3 ROS描述Apache-2.0；RevoLab代码MIT，资产/第三方各查许可 | **RevoLab v1冻结20261014资产**，明确旧v0权重/抓取缓存不兼容v1，且未发布v1预训练权重或验证的实机部署结果。RevoSim触觉采样点不等于硬件独立传感通道 [B3–B7] |

研究采用的可核验例子：

- **Shadow**：OpenAI《Learning Dexterous In-Hand Manipulation》摘要明确在真实Shadow Hand上完成视觉物体重定向；训练采用仿真域随机化。它证明研究控制链可用，不证明触控板电容兼容 [S2]。
- **Allegro**：HORA（CoRL 2022）官方代码说明主要开发用内部Allegro版本，另提供公开**V4**实现及视频。因此不能把全部论文数字当作V4商品默认性能 [A5]。
- **LEAP v1**：RSS 2023公开项目有模型、实机遥操作、视频学习与sim-to-real代码；另外MuJoCo Playground纳入LEAP任务，提供独立软件生态证据。v2与Advanced不继承这些标定和结果 [L1][L8][L9]。
- **新款S/Revo3/Dex5/V6 F**：目前更多证据来自厂家文件、代码及演示；长期外部用户可靠性、低力触碰质量和维修体验仍需独立验证。

## 4. 对触控板最有用的选型标准

要完成长程Mac操作，手首先要能稳定生成接触轨迹。手指总数没有“5必然优于4”的结论；当前最关键的是选出的两根手指有足够独立运动能力。

1. **两指能分开贴面、抬起、滑动并改变间距。** 独立侧摆和各段屈伸可减少腕部代偿。LEAP v1、Allegro四指、Tesollo20DoF、Revo3较适合；Inspire/Revo2/Dex3仍可测试拇指配另一指，但相邻两指方式受到更强约束。
2. **能控制轻触，而不仅是抓得很牢。** 关节电流到指尖力的映射会随姿态、齿隙和摩擦变化；传感分辨率与控制分辨率不同。应测低速爬行、静摩擦起步、抬指延迟、触面力峰值和15分钟温漂。
3. **指尖触感与电学都要合适。** 普通硅胶、TPU或塑料表皮可能不能被电容触控板可靠检测。导电涂层/柔性导电指套、接地/电容耦合和有效接触面积需要实物试验。**不能把“MuJoCo发生接触”当成“Mac收到触摸”。** 修改表皮还会影响摩擦、力传感和质保。
4. **两指能进入160×110 mm的实际工作区域。** 整手宽度不是判据，需用指尖工作空间、掌部碰撞与腕姿联合评估。表中尺寸只是厂家外形，不是已通过触控板覆盖测试。
5. **人形集成看整条手腕链。** 需确认法兰、重心/惯量、腕部持续扭矩、供电峰值、线缆回转和控制总线。两台都用EtherCAT或24V不代表机械/协议即插即用。

## 5. 建议的下一轮比较与采购候选

**马上进入仿真对比的三项：**

- **LEAP v1**：第一新增候选。模型和控制生态成熟，手指侧摆清楚，也有可承担的实物组装价参考。先用同一无学习控制器检验指尖可达和稳定滑动，再决定是否训练。
- **Tesollo DG-5F-S 20-DoF**：第二新增候选。模型文件已存在、20独立关节、880g，有机会同时服务触控板与未来人形路线；优先核验具体模型参数，不急着购买。
- **现有 Allegro V3**：保留对照。先纠正安装、轴、接触和执行器设定；只有同条件跑差才可归因于构型不适合。Shadow E3M5可作为补充高自由度参考，不必同时扩成第四条主线。

**未来实物的三项优先候选：**

- **LEAP v1 Full组装手**：低预算且愿意改指尖/自己维护时优先。正式核对电机版本、软指尖、电源、通讯板、备件与单手价格。
- **Tesollo DG-5F-S 20-DoF**：目标偏完整人形与独立关节操作时优先。询价时要求确切20DoF/传感器配置、SDK控制模式、低力性能和腕部图纸。
- **BrainCo Revo3 U21F或U21T**：构型与力/触觉反馈值得评估；先取得重量、外形CAD、轻触演示、交付型号与完整报价。若未来底座明确为G1，再把 **Unitree Dex5-1/P** 与其做同条件对比。

采购前应让候选完成同一短演示：单指拖动→两指独立抬落→双指滚动→连续变间距→变间距同时横移→15分钟连续重复。记录真实触摸识别、力峰值、接触丢失、指尖温度/磨损与SDK延迟。不要求一开始禁用拇指；食/中指与拇指/另一指是两个实验条件。

上述仿真导入、场景调整和运行应由 **Kimi** 按项目约定实施。此报告只提供选型与验收依据，没有替实施任务宣告运行成功。

## 来源索引

[来源、版本与价格证据](2026-10-31-hand-candidates-evidence.json)记录URL、访问时间、简短原句、未知项及源文件哈希。原始网页和手册快照保留在私有核验档案，不随公开仓库分发。访问失败或只有JS外壳的页面未作为价格来源。

- [L1] [LEAP v1官方项目](https://v1.leaphand.com/)；[L2] [v1零件/BOM](https://v1.leaphand.com/parts)；[L7] [v1购买页](https://v1.leaphand.com/purchase)。
- [L3] [LEAP v2 Low-cost](https://v2.leaphand.com/)；[L4] [v2零件与组装价](https://v2.leaphand.com/parts)。
- [L5] [LEAP v2 Advanced](https://v2-adv.leaphand.com/)；[L6] [Advanced零件/BOM](https://v2-adv.leaphand.com/parts)。
- [L8] [Menagerie LEAP模型说明（冻结版本）](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/leap_hand/README.md)；[L9] [MuJoCo Playground LEAP环境](https://github.com/google-deepmind/mujoco_playground/tree/12ad7f3842791ced39f27e5ac96d066bbe361653/mujoco_playground/_src/manipulation/leap_hand)。
- [A1] [Allegro V4](https://www.allegrohand.com/sub/product/p.php?idx=1)；[A2] [V5 Plus](https://www.allegrohand.com/sub/product/p.php?idx=4)；[A3] [V6 F](https://www.allegrohand.com/sub/product/p.php?idx=25)。
- [A6] [Allegro官方型号列表（V5为9DoF）](https://www.allegrohand.com/)。
- [A4] [现有Menagerie Allegro V3模型说明](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/wonik_allegro/README.md)；[A5] [HORA官方代码与版本说明](https://github.com/HaozhiQi/hora)。
- [I1] [Inspire RH56DFX](https://www.inspire-robots.com/dexterous%20hands/rh56dfx-series/)；[I2] [当前RH56E2页面（网址仍含DFTP）](https://www.inspire-robots.com/dexterous%20hands/rh56dftp-series/)。
- [U1] [Unitree Dex3-1](https://www.unitree.com/Dex3-1/)；[U2] [Dex5-1/P](https://www.unitree.com/Dex5-1/)；[U3] [官方XR遥操作与G1/Inspire支持](https://github.com/unitreerobotics/xr_teleoperate/tree/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b)；[U4] [Unitree Isaac Lab任务](https://github.com/unitreerobotics/unitree_sim_isaaclab/tree/e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc)。
- [T1] [Tesollo DG-5F-S](https://www.tesollo.com/products/grippers/dg-5f-s)；[T2] [DG-5F-M](https://www.tesollo.com/products/grippers/dg-5f-m)；[T3] [官方模型与BSD许可](https://github.com/tesollodelto/tesollo_model/tree/84be60a8e06d22fbc00aec3534011ed0e5008325)；[T4] [S模型与15/20DoF分型](https://github.com/tesollodelto/tesollo_model/blob/84be60a8e06d22fbc00aec3534011ed0e5008325/dg5fs/README.md)。
- [B1] [Revo2官方产品手册](https://www.brainco-hz.com/docs/revolimb-hand/revo2/parameters.html)；[B2] [Revo3官方产品手册](https://www.brainco-hz.com/docs/revolimb-hand/revo3/parameters.html)。
- [B3] [Revo2官方URDF](https://github.com/BrainCoTech/revo2_description)；[B4] [Revo3官方ROS2描述/控制](https://github.com/BrainCoTech/brainco_revo3_ros2)；[B5] [RevoLab版本与验证边界](https://github.com/BrainCoTech/RevoLab/blob/0fb72da54c2401638f808eb814bbfe55b6f9931f/README.md)；[B6] [Revo-MjLab](https://github.com/BrainCoTech/Revo-MjLab/blob/2727426b51036daf1a0353857cf0a6f0704928b8/README.md)；[B7] [RevoSim触觉仿真](https://github.com/BrainCoTech/RevoSim/blob/c862db29a964422721e231533a80078181d381a5/README.md)。
- [S1] [Shadow官方系列对比](https://shadowrobot.com/dexterous-hand-series/)；[S2] [OpenAI 2018实机Shadow研究](https://arxiv.org/abs/1808.00177)；[S3] [Menagerie Shadow E3M5模型](https://github.com/google-deepmind/mujoco_menagerie/blob/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/shadow_hand/README.md)。
