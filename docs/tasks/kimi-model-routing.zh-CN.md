# Kimi 执行模型与统一调用

2026-11-01，用户明确选择 **Kimi K2.8 Preview**，并要求其他 GPT/Codex 对话调用时沿用。协调者负责架构、预算、独立验收和文档/Git；具体实现、部署、训练与直播仍由 Kimi 执行。

本机共享 `kimi-handoff` runner 默认显式传入 CLI alias `kimi-code/kimi-for-coding`。官方 API model ID 为 `kimi-for-coding`；官方当前将其映射为 K2.8 Preview。旧的 `kimi-for-coding-highspeed` 映射为 K2.7 Code Highspeed。使用原生 CLI，避免旧项目脚本强制覆盖模型。新直接 CLI 会话也默认选用 2.8；已经运行的会话不变，升级任务使用简短检查点开启新会话。

共享设置作用于同一主机上使用该入口的后续对话，不自动改写其他主机、云端工具或显式模型覆盖。回执记录请求模型；模型自述不作为路由证据。保留现有会员登录和计费入口，不创建新付费 API、不改加油包或消费上限。

## 价格参考

以下为每百万 token 的美元价格，核验日期为 2026-11-01：

| 模型 | 未缓存输入 | 缓存读取 | 输出 | 来源 |
|---|---:|---:|---:|---|
| K2.7 Code | $0.95 | $0.19 | $4.00 | 官方开放平台 |
| K2.7 Code Highspeed | $1.90 | $0.38 | $8.00 | 官方开放平台 |
| K2.8 Preview | $1.00 | $0.25 | $4.00 | LLM Gateway 所列 Tencent 路由 |

2.8 报价来自第三方，不是已核实的 Kimi 会员扣额费率。对相同历史 token 构成重算，2.8 参考价比旧高速版低约 37.88%，比普通 2.7 高约 24.25%；这不是实际账单、实际额度节省或相同任务的模型性能对照。用户提及近期 2.8 折扣，未核实折扣率和有效期，因此不纳入数值估算。

验证结果：默认/显式覆盖/非法 alias 三项路由检查通过；一次新会话实际返回只读样例随机值，13.2 秒完成，文件散列未变化，用量记录选中 `kimi-code/kimi-for-coding`。CLI 配置检查与技能校验通过。原始会话与本机配置私有留存。

来源：[官方模型映射](https://www.kimi.com/code/docs/kimi-code/models.html)、[官方 API 价格](https://platform.kimi.ai/docs/pricing/chat)、[会员额度规则](https://www.kimi.com/code/docs/kimi-code/membership.html)、[2.8 第三方报价](https://llmgateway.io/models/kimi-k2.8-preview)。
