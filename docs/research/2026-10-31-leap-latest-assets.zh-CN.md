# LEAP 最新高配置路线：版本、仿真资产与后续 DIY 核查

核查时间：2026-10-31（香港时间）。范围：官方公开网站、GitHub API/文件、CAD下载；没有安装环境、运行仿真、提交表单、联系厂商或实施硬件操作。原始来源快照保留在私有审计目录，公开[证据清单](2026-10-31-leap-latest-assets-evidence.json)；模型接入与运行成绩另见 STATUS.md。

## 结论

**本次主线应改为 LEAP Hand v2 Advanced，五指、17个马达、21个运动关节；后续DIY以金属手掌/齿轮为候选配置。** 官方真实URDF、STL、完整STEP、分件STEP/3MF都可找到，已经实际下载并解析了左右URDF，所以没有理由仅因为v1已有MuJoCo Menagerie就回退v1。

但“最新高配”不能写成“现成高精度仿真模型”。Advanced官方模型有可修正的关节耦合、辅助link质量和网格路径问题；它的软体腱驱动也需要近似、辨识与验证。金属版官方还明确提示MCP前屈方向可能有10–15°回差。软件阶段可以继续，采购和精度承诺要以未来实测为准。

此次检索官方账号全部9个公开仓库及v1/v2官网，看到的硬件新支线为 **v2 Advanced** 和 **v2 Low-cost**，未找到官方v3。API提交日期不能拿来代表硬件代际：v1 API最后提交比Advanced更晚，但仍然是v1硬件。

## 版本不能混叫

| 项目 | v2 Advanced（本次主线） | v2 Low-cost | v1 Full（对照，不替代主线） |
|---|---|---|---|
| 手指 | 5指 | 4指 | 4指 |
| 运动关节/DoF | 21 | 16 | 16 |
| 独立马达数 | 17 | 8 | 16 |
| 驱动结构 | 四根非拇指各MCP侧摆、MCP前屈、PIP/DIP共用curl；拇指3马达；手掌2马达 | 每指MCP侧摆+一个卷曲腱，卷曲关联MCP/PIP/DIP | 16个关节独立驱动 |
| 手掌 | 两个独立驱动的折叠关节 | 一体手掌，top/wrist mount可选 | 刚性手掌 |
| 主要电机 | 6×XM430-W210-T、6×XM430-W350-T、5×XC330-T288-T | 4×Feetech HLS3612M、4×HLS3606M | 16×XC330-M288-T；Lite才是XL330-M288-T |
| 指体 | TPU+PLA软硬混合，腱驱动 | TPU+PLA软硬混合，更强欠驱动性 | 刚性关节主体，可配软指尖 |
| 软件控制维数 | 17D direct motor接口；20D人手姿态输入经过retargeting，不能当20个独立硬件自由度 | 8D，侧摆rad、卷曲归一化0–1 | 16D |
| 可用模型 | 官方左右URDF+STL，另有CAD；本次未找到原生MJCF | CAD存在；官网“URDF”实际误链Advanced，不能当作本型号已就绪 | 官方左右URDF，Isaac Gym/Isaac Lab生态；另有第三方Menagerie |
| 本次价格依据 | 官方Dec 2024 BOM：Plastic US$4,079.88；Metal US$5,079.88 | 官网约US$300零件、有限现货组装约US$500；整包物料表另列US$221.95+251.53，可做多手共享物料 | 官网Full under US$2,000 |

这些价格是网页公开参考和有日期的BOM，不是2026年供货报价；税费、运费、工具、打印机、装配、加工返工和附加触觉传感器不应混入“标价即落地总成本”的说法。Advanced首页US$3000、parts导语US$4000与详细BOM互不一致，预算采用标有日期的详细BOM，并保留更新时间。

## 官方仓库和固定版本

| 官方仓库 | 当前分支/固定commit | commit时间UTC | 资产位置 |
|---|---|---|---|
| [Advanced API](https://github.com/leap-hand/LEAP_Hand_V2_Adv_API) | main / `a0936196e8fa0bcefff045d6fc02523d6da591f2` | 2025-06-13 17:36:35 | `leap_v2/leap_v2/leap_v2_left/robot.urdf` 与 `...right/robot.urdf`；各自同目录STL |
| [Low-cost API](https://github.com/leap-hand/LEAP_Hand_V2_API) | main / `eec4bd58274484271cdfc33d7c5a937b5c936f9b` | 2025-09-01 21:37:07 | 控制API；完整递归tree内无URDF/MJCF |
| [v1 API](https://github.com/leap-hand/LEAP_Hand_API) | main / `b0d00c881d0119077b2cab771a6de44f5aaec904` | 2025-10-20 16:31:10 | 控制API；模型另行发布 |
| [v1 Isaac Gym](https://github.com/leap-hand/LEAP_Hand_Sim) | master / `150bc3d4b61fd6619193ba5a8ef209f3609ced89` | 2024-05-18 18:48:47 | `assets/leap_hand/robot.urdf` |
| [v1 Isaac Lab](https://github.com/leap-hand/LEAP_Hand_Isaac_Lab) | main / `c576e6a83183ba504aef8f57015f2899b799db1a` | 2025-09-04 21:16:11 | `source/LEAP_Isaaclab/LEAP_Isaaclab/assets/leap_hand_v1_right/` 下USD及physics配置 |

“当前”是本次请求返回的默认分支头，不代表版本被官方做过完整测试。所有repo tree响应均保存；Advanced和Low-cost trees未被GitHub截断。

### Advanced 下载入口

- [官网](https://v2-adv.leaphand.com/)、[CAD目录](https://v2-adv.leaphand.com/leap_cad)、[BOM](https://v2-adv.leaphand.com/parts)。
- [固定左手URDF](https://raw.githubusercontent.com/leap-hand/LEAP_Hand_V2_Adv_API/a0936196e8fa0bcefff045d6fc02523d6da591f2/leap_v2/leap_v2/leap_v2_left/robot.urdf)。
- [固定右手URDF](https://raw.githubusercontent.com/leap-hand/LEAP_Hand_V2_Adv_API/a0936196e8fa0bcefff045d6fc02523d6da591f2/leap_v2/leap_v2/leap_v2_right/robot.urdf)。
- 每侧10个唯一引用STL，左约20.60MB、右约20.90MB；全部文件名都能在对应repo目录找到。固定 URL 逐项写在配套证据清单。
- [官网左URDF ZIP](https://v2-adv.leaphand.com/static/cad/leap_v2_adv_left.zip)、[官网右URDF ZIP](https://v2-adv.leaphand.com/static/cad/leap_v2_adv_right.zip)。本次已完整下载（约8.46MB/15.03MB），ZIP内URDF已解析，STL可在ZIP内找到。
- [完整左手STEP](https://v2-adv.leaphand.com/static/cad/full_hand/leap_v2_full_assembly_left.stp)、[完整右手STEP](https://v2-adv.leaphand.com/static/cad/full_hand/leap_v2_full_assembly_right.stp)。本次左文件请求成功，但超过32MiB读取预算后停止，不能说已完整校验装配CAD。
- [金属四指下掌STEP](https://v2-adv.leaphand.com/static/cad/palm_parts/four_fingers_lower_palm_metal.stp)已下载539,800字节。CAD目录还有对应PLA件、metal/PLA gears、MCP件、手指/拇指多材料STEP和打印3MF。

### Advanced 不能直接照搬进动力学仿真的问题

**1. 官网ZIP与GitHub是两套不同快照。**

| 已解析结果 | GitHub左 | GitHub右 | 官网左ZIP | 官网右ZIP |
|---|---:|---:|---:|---:|
| links | 32 | 32 | 22 | 22 |
| revolute | 21 | 21 | 21 | 21 |
| fixed | 10 | 10 | 0 | 0 |
| mimic | 4 | 0 | 0 | 0 |
| inertial质量合计kg | 2.505 | 2.505 | 1.505 | 1.505 |

GitHub多了`realtip*`和`realdip*`这10个辅助标记，每个附带0.1kg惯性，合计增加1kg。剔除它们后的本体质量和官网一致。对动力学模型应把标记变成MuJoCo site，保存坐标而不增加质量/碰撞；不能把2.505kg当作真实手重。1.505kg也只是URDF参数，尚未证实是特定metal/plastic实物实测。

官网ZIP使用`thumb_with_motor_base`等名称，GitHub使用`thumb_with_motor`；右ZIP还有`4_finger_frame_mirror_simple.stl`。不得混用一套URDF与另一套网格。GitHub模型的IK标记便利、commit可固定；官网ZIP可用SHA256固定。选一套作为来源，再显式记录所有适配。

**2. 关节耦合存在缺失和错误。** 左URDF中`middle_pip`的mimic目标写成`index_dip`，与官方每指独立curl接口不一致；右URDF完全没有mimic。不能由“成功加载21关节”推断21个独立动作维度。4根非拇指PIP/DIP共用各自一个腱，需要17维驱动映射。

阶段A可采用每指`PIP=DIP`的刚体等式近似，并记录这是近似。软体指骨实际弯曲分配会受到接触、材料刚度和腱张力影响；自由空间等角mimic不等于接触下真实腱驱动。后续要让弹性、阻尼、腱长度/张力与关节分配可校准，不把简单等式包装成完整软体仿真。

**3. 路径与参数需审查。** 网格URL在URDF中写成`package:///basename.stl`，缺具体ROS包名；导入时要改为本地下载路径。URDF包含inertial、joint limit，但没有transmission。普遍0.5Nm的effort、8rad/s的velocity不能直接当作这17个不同电机的经验证性能曲线。

**4. 安全下载白名单。** 上游`config.json`包含credential-like字段；它们与仿真无关，不使用、不复制到项目或公开Git。`.part`、`__pycache__`、对齐CSV、导出配置亦不需要。只取选定URDF、引用STL、适用LICENSE/归属信息，并记录原始hash。

### 17维接口建议（给Kimi实现）

按官方direct motor order保留统一动作向量：

```text
0 index_mcps       1 index_mcpf       2 index_curl
3 middle_mcps      4 middle_mcpf      5 middle_curl
6 ring_mcps        7 ring_mcpf        8 ring_curl
9 pinky_mcps      10 pinky_mcpf      11 pinky_curl
12 thumb_mcps     13 thumb_mcpf      14 thumb_ip
15 palm_thumb     16 palm_4_finger
```

`curl`是一个动作通道，映射到本指PIP/DIP。不要以URDF出现顺序作为动作token顺序；左右手URDF顺序不同。20D glove输入是人手retargeting输入，17D motor输出才是硬件执行接口；两个手掌马达也必须保留，不应悄悄冻结。

## Metal/Plastic 与 DIY现实

Advanced官网推荐金属手掌和齿轮以提高耐久性，额外BOM约US$1000；软硬混合手指仍是多材料打印，并不是整只手全金属。公开CAD里有metal/PLA零件对，但本次未找到分开标定的metal/plastic URDF、惯性表或接触参数集。

CAD页明确写了金属打印MCP前屈可能有 **10–15°回差**，部分加工需要锉修，建议对齿轮和MCP内外件考虑更精密的CNC方案。因而建议定义为“Advanced结构 + 金属耐久性目标”，先完成模型/控制和触控板任务，再依据实际误差预算确定加工路线。

手指的低硬度TPU/PLA多材料打印依赖工艺。官网提供Snapmaker J1S配套设置；普通单喷嘴Bambu AMS不能自动完成这种TPU切换，官网也提示双喷嘴设备仍受极软TPU可打印性影响。打印机、喷嘴、耗材、腱线、装配校准和维修不应被一个零件总价遮蔽。

## 传感器究竟有什么

- **Advanced**：Dynamixel马达位置、速度、电流反馈路径存在，`pos_srv/vel_srv/eff_srv`调用`read_pos/read_vel/read_cur`。`effort`字段实际读取电流，不是校准的指尖接触力。README说有ROS2服务，但当前node内对应`create_service`被注释，因此不能宣称开箱即用ROS状态服务已验收。
- **Low-cost**：README明确有位置、速度、马达电流读取，建议位置+速度联合读取最高约90Hz。腱马达位置不等于每个柔性关节的独立角度观测。README建议7.6V供电，但官网BOM写12V17A电源，硬件阶段必须核对具体电机额定值与供电设定，不能直接照两段不一致文档接线。
- **v1 Full**：Dynamixel控制生态提供位置/速度/电流等本体反馈；Lite的电机规格不同，不能把Full的电流/力矩性能套到Lite。
- Advanced与Low-cost CAD提供`touch`指尖、硬插片和走线槽，是**安装传感器的结构选项**；未证实标准BOM含某款触觉阵列、独立腱张力传感器或Mac电容触控兼容指尖。

未来真实触控板还有导电性、接地/电容耦合、接触面积、法向力及多指识别问题；当前“能触碰平面”的仿真不构成Mac真实触控验收。

## Low-cost和v1资产的核查边界

Low-cost的[CAD页](https://v2.leaphand.com/leap_cad)确有圆指/平指、top/wrist mount完整装配STEP、零件STEP/3MF与Onshape链接。本次实际下载了[右手平指top mount STEP](https://v2.leaphand.com/static/cad/assembly/flat_finger_top_mount_right.stp)（15,789,020字节）。但页面的“Left/Right URDF Files”链接都指向 **Advanced五指仓库**；Low-cost自己的API tree没有URDF。应记录“CAD可用、正确型号仿真模型尚未核实”，不把错误链接当完成。

v1的[官方右URDF ZIP](https://v1.leaphand.com/static/cad/urdf/leap_right.zip)已实际下载2,950,694字节，里面`robot.urdf`解析为17 links、16 revolute、0 mimic，符合其独立16马达结构。官方[CAD页](https://v1.leaphand.com/leap_cad)还有左右完整STEP、可更换指尖与安装件。Menagerie/Playground属于v1路线，不能改名字充作v2；此前Playground配置标XL330，也不能自动称为XC330 Full已标定。

## 许可证逐来源保留

| 来源 | 本次看到的许可文本 | 使用结论 |
|---|---|---|
| Advanced `leap_v2/LICENSE` | Apache-2.0全文；无repo根统一LICENSE；package.xml仍写TODO | 可确认存在嵌套软件license，不能因此把CAD统一MIT化 |
| Advanced官网CAD页/下载页 | CC BY-NC-SA，非商业/署名/相同方式共享 | 按此保留CAD来源；本项目开源许可证不覆盖上游CAD |
| Low-cost API README | 自有软件MIT、Feetech部分另有许可；CAD文字仍说CC BY-NC-SA | 与当前CAD页有版本差异，逐资产记载，不能只选更宽松的一段 |
| Low-cost当前CAD请求页及CAD下载页 | CC BY-SA 4.0，含商业使用 | 当前网站CAD文本明确，但与repo旧README冲突需保留证据 |
| v1 API LICENSE | CC BY-NC 4.0 | 不是所有LEAP代码统一MIT |
| v1官方Sim LICENSE.txt | MIT | 软件仓库license与官网CAD来源分开记 |
| v1当前官网CAD请求页及下载页 | CC BY-SA 4.0，含商业使用 | 与v1 API的CC BY-NC不同，不能混为一项 |

阶段A实施可采用“上游资产下载到gitignored缓存 + 公布固定URL/hash、许可证说明、项目自有转换/修补逻辑”。不需要把几十MB网格带进主仓库才能继续CPU仿真。对于本身源于上游CAD的转换产物，仍要保留来源和对应许可，不把转格式当消除许可。

本次只读取公开页面与直接下载，未向表单提交姓名、邮箱、组织或接受交互式条款；下载页自身明确附带许可告知，报告也不将其描述为无条件资产。

## 本轮验收与下一步

本轮已完成：官方资产查找、commit固定、实际文件下载、XML解析、路径核对、版本/质量/耦合差异发现、价格与许可证据保存。**没有仿真加载、GPU执行、策略训练或真实触控板成绩**。

Kimi阶段A按已派发的有界任务取一侧Advanced、修正mimic和marker质量、加入17D驱动映射、CPU加载并生成真实渲染/动作证据。手型明确后再做工作区、双指独立侧摆/间距、点按/滑动/捏合、接触稳定性和预算测试；通过后再进H800批量训练。v1仅保留参照实现与对比基线。

## 固定下载内容的SHA256

官网ZIP不带Git commit，下面用实际字节SHA256冻结；GitHub URDF另有已记录的commit。

| 来源 | 文件 | SHA256 |
|---|---|---|
| github_raw | [leap_latest_adv_left_urdf.txt](https://raw.githubusercontent.com/leap-hand/LEAP_Hand_V2_Adv_API/a0936196e8fa0bcefff045d6fc02523d6da591f2/leap_v2/leap_v2/leap_v2_left/robot.urdf) | `2c58f4e5bc891aba47dad9414553023e0112b4a3c2d3a888b1c9b3005c442f98` |
| github_raw | [leap_latest_adv_right_urdf.txt](https://raw.githubusercontent.com/leap-hand/LEAP_Hand_V2_Adv_API/a0936196e8fa0bcefff045d6fc02523d6da591f2/leap_v2/leap_v2/leap_v2_right/robot.urdf) | `cdd886d2f61dcb2904bbf2e9a0b7401f699ceb62521bf7512687dc880997230e` |
| official_website_download | [leap_latest_adv_zip_left.zip](https://v2-adv.leaphand.com/static/cad/leap_v2_adv_left.zip) | `e899e18575a9fb2c3e0110e0123dfb7c38b5561687a84037d2e3092f7fc299c2` |
| official_website_download | [leap_latest_adv_zip_right.zip](https://v2-adv.leaphand.com/static/cad/leap_v2_adv_right.zip) | `3f349c7c81cb7fa0c2cc27b10290a81d924cb2e15b9f29859a39b9712c0f3975` |
