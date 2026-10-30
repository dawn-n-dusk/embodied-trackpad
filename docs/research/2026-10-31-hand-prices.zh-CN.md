# 灵巧手公开价格补充核验

> 价格更新：本页保留首轮研究口径。新增国内媒体采访、商家挂牌、历史与二手检索结果见[市场价格补充](2026-10-31-hand-market-prices.zh-CN.md)；其中已补齐 Revo、Allegro、Tesollo S 等公开预算，不再仅以官方是否列价判断价格未知。

访问日期：2026-10-31，香港时间。调查窗口约01:06—01:14，仅阅读公开页面；未联系商家、提交询价、加入购物车或下单。本补充与[主报告](2026-10-31-hand-candidates.zh-CN.md)一起使用，新增型号不能与旧款混价。

新增结果包括：**Unitree官方新款Dex5-S公开标价，以及Inspire RH56DFX、Tesollo DG-5F的公开零售参考。** 单/双手、附件或授权身份未充分确定的部分明确保留，不把它们当成最终到手价。

## 官方商店价格

| 产品与变体 | 标价 | 单位、税运、供货边界 |
|---|---|---|
| **Unitree Dex5-S**，variant `67654587154665` | **US$6,350** | 官方产品页与公开产品数据一致。页面没有明确“一只/一对”计价，也没有左右手选择，不能擅自写成单手价。不是人形机器人整机套餐；但完整装箱清单未核得。产品页明确 **“Shippment will begin in December”**，并写backordered，属于预售/待发货；不能标成现货。关税不含，运费在结账计算，销售税口径未确认。 |
| **Unitree Dex5-S Pro**，variant `67654671204585` | 公开变体数据 **US$13,500** | 商品页要求联系销售购买Pro，公开数据为不可直接购买状态。单手/一对与装箱同样未确认；不是成交报价。 |

来源：[Unitree官方商品页](https://shop.unitree.com/products/unitree-g1-%E5%89%AF%E6%9C%AC)、[官方产品数据](https://shop.unitree.com/products.json?limit=250)、[运费政策](https://shop.unitree.com/policies/shipping-policy)。商品旧URL含“G1-副本”，但标题与产品数据均为Dex5-S。

**Dex5-S是本次新发现的SKU。** 商店描述为22DoF/手，不能将这两笔价格套到原报告的Dex5-1或Dex5-1P，也不能把Dex5-1/P的16主动+4耦合、重量或触觉配置搬过来。S的完整硬件与仿真资产仍需独立调查。

## 公开零售参考：未独立核实品牌授权关系

下列是专业机器人零售商自己的公开商品页。已核对具体商品/部分SKU，但本次没有独立取得厂家对其该品牌授权的证明，因此明确标作**零售参考**，不称“厂家授权报价”。

| 商家/产品 | 公开价格 | SKU、单位与附件 | 税运/供货 |
|---|---|---|---|
| Génération Robots — **Inspire RH56DFX，Left Side** | **€4,150未税 / €4,980含税** | 搜索页结构化商品记录：`404369-2020`，Left Side，参考号`A-000000-07836`，quantity=1；属于左手SKU，不是一对。商品详情另有通用产品号`A-000000-07816`，并列左/右选择；未通过变体操作验证右手同价。电源、通讯盒和法兰是否包含未明确。 | 默认欧元页面列VAT含/不含两价，不代表香港进口到手成本。运费未核得；页面写2—3周交付，不能保证现货。 |
| MYBOTSHOP — **Tesollo Delto Gripper DG-5F** | **€18,886.95含19%德国VAT，运费另计** | 产品号`MBS-G-45`，厂家型号`DG-5F`；列DG-5F / Short Wrist选择，但价格与最终变体映射未完全确定。标准装箱明确：**1×DG-5F手、1×24V电源、1×以太网与控制线、1×安装法兰**，因此是一套单手套装。左右手未明确。 | 页面写需订货、订货后10天可交付，不代表实际库存或承诺。页面规格1.76kg、20DoF，属于较大DG-5F型号；**不能当作880g DG-5F-S价格，也不能据官网60%比例推算S售价**。旧SKU与当前DG-5F-M具体硬件修订仍需确认。 |

来源：[RH56DFX商品页](https://www.generationrobots.com/en/404369-rh56dfx-robotic-hand-2020.html)、[Inspire搜索页的Left Side记录](https://www.generationrobots.com/en/search?controller=search&s=Inspire)、[Tesollo DG-5F商品页](https://www.mybotshop.de/Tesollo-Delto-Gripper-DG-5F_1)。

## 已进一步核验，但仍没有金额

- **Allegro**：厂家[官方销售网络](https://www.allegrohand.com/sub/contact/network.php)明确列MYBOTSHOP为德国/奥地利/瑞士经销渠道。其[V4右手商品页](https://www.mybotshop.de/Allegro-Hand-V4-rechts)具体为`MBS-G-33`，明确 **“Preis auf Anfrage”**（询价）；搜索页V4左/右、V5、V5 Plus有/无指尖传感、V5 Sense、V6均没有可核验整手金额。另有指尖配件价格，但不能当成整手价格。
- **Unitree Dex3-1、Dex5-1/P**：本次没有在官方商店核得独立、对应型号的金额。Dex5-S价格不能代替它们。
- **BrainCo Revo2/Revo3**：没有核得独立手的可信公开价。Génération Robots有含Revo2的Unitree整机套餐，但不能用套餐价或套餐差额推算单手价格。
- **Tesollo DG-5F-S**：仍无已确认的公开金额；上面的DG-5F零售价不是S价格。

RobotShop查询返回403，未将访问失败写成“没有价格”；部分商店搜索会返回无关商品，已排除。以上没有任何一项经过结账或厂家书面确认。
