# 训练监看（training monitor，观察端）

2026-10-31：接口和有限 TEST 来源的浏览器检查已审查，验证服务已经结束。
当前没有接入实际训练器，不能将本接口称为已交付的训练过程机器人推流。
验收及覆盖范围见[实验记录](../experiments/records/training-monitor-interface-20261031.json)。

本接口让训练器/渲染器把**最新一帧真实画面 + 版本化元数据**发布到运行目录，本地查看器在
`127.0.0.1` 上如实呈现。它是纯观察者：不仿真、不产生动作、不以脚本演示冒充训练。
所有阶段标签均为**生产者上报，未经独立验收**（`producer_reported_pending_independent_acceptance`）。

## 组成

- `src/embodied_trackpad/monitor/protocol.py` — 记录 schema、严格校验、显示状态推导、
  带验证的运行目录读取器。无效数据一律失败关闭（`MonitorProtocolError`）。
- `src/embodied_trackpad/monitor/publisher.py` — 生产者侧原子发布器：tmp+fsync+`os.replace`，
  存储有界（仅最新帧 + 最新记录 + 静态 metadata + 上限日志），`frame_seq` 单调递增。
- `tools/training_live_view.py` — 仅 GET 的本地 HTTP 查看器（默认端口 8770 起，向后探测）。

## 运行目录线协议

`latest.json`（schema `training-monitor/1`）必填字段：`run_id`、`source_kind`
（`training_environment | checkpoint_evaluation | scripted_diagnostic | test_fixture`）、
`lifecycle`（`waiting | training | evaluating | stalled | failed | completed`）、
`frame_seq`、`physics_step`、`sim_time_s`、`source_utc`、`producer`（`heartbeat_unix`、`progress`）、
`train_step` / `gradient_updates`（缺测为 null，绝不虚构 0）、`world_id`、`environment_count`、
`policy_kind`、`checkpoint`（`policy_kind=learned` 时必填）、`frame`（null 或固定名
`frame.png` + sha256 + 宽高 + `rendered_unix`）。

读取端边界：记录/帧读取有大小上限（1 MiB / 64 MiB）；PNG IHDR 宽高必须与 JSON 元数据一致；
逃逸运行目录的符号链接被拒绝；生产者心跳/渲染时间戳不得超前读取端时钟超过 5 秒
（`CLOCK_TOLERANCE_S`，超前的未来时间戳被拒绝，绝不按 0 秒新鲜度截断）。

## 显示状态优先级

`FAILED`/`ENDED`（终态优先）→ `REPLAY`（viewer 启动前已存在的历史记录）→
`WAITING`（无帧或声明 waiting）→ `STALLED`（声明 stalled）→
`TEST`/`DEMO`（fixture/脚本来源新鲜时；超时降级 STALLED；**绝不 LIVE、绝不训练中**）→
仅真实、有帧、心跳与画面均新鲜的来源才能 `LIVE`。
`training_declared` 是生产者侧合取；`training_active` 额外要求当前为新鲜 LIVE 画面
（非停滞/非历史/非终态）。fixture/scripted 始终带 TEST/DEMO 徽章。

浏览器端帧身份为 `run_id + frame_seq`（`X-Run-Id` / `X-Frame-Seq` 响应头），不是像素哈希：
重新渲染的相同画面在新序号下仍会解码刷新；同身份不刷新。图像元素在首帧解码完成前保持隐藏
（无破图图标）。看门狗的解码帧新鲜度同样约束 TEST/DEMO：新鲜时保持来源状态与徽章
（绝不升级为 LIVE），无新解码帧即降级 STALLED。元数据轮询同样单发、有界。

## 查看器生命周期

```
PYTHONPATH=src python tools/training_live_view.py \
    --run-dir <运行目录> --port 8770 --max-wall-s 600 \
    --out-dir .local/training-monitor/server
```

- `--max-wall-s` 为**声明寿命**：有限，默认 600 s；显式声明最高 86400 s，
  必须与已授权的任务预算一致。无来源启动同样受该期限约束。
- 验证到终态（completed/failed）后，服务在最终画面宽限期（`TERMINAL_GRACE_S`，30 s）后自动停止。
- 到期或 SIGTERM/SIGINT 释放端口，并把停止回执写入 `result.json`
  （含 `stop_reason`：`deadline_expired | terminal_state | signal | error`）；
  启动信息写入 `server.json` / `viewer.pid`。
- 同一 `run_id` 内 `frame_seq` 必须严格递增：序号回退或同序号内容变异均被拒绝并保留
  最后已验证状态；新 `run_id` 重置序号身份。
