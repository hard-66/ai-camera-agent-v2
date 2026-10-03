# JUDGE-001 真实调用验证记录（Typesafe Jev）

日期：2026-10-03。验证人：Jev/判断组。
所有请求均为真实 provider 请求；本文件不含任何 Key。

## 结论

**TypesafeJevBackend 真实调用 PASS**：官方端点 `POST https://api.typesafe.ai/v1/systemone`，
Bearer 认证，模型 `jev-1.13.0`，请求/响应格式与 `agent_system/judge.py` 的
`build_jev_payload` / `parse_jev_response` 逐字段一致，无需修改。

## 证据 1：最小 ping（教程"10 秒自测"原样执行）

请求体：`{"state":"ping","model":"jev-latest","questions":{"ok":{"type":"noul","instructions":"Is this a test?"}}}`

```
{"model":"jev-1.13.0","answers":{"ok":{"type":"noul","noul":0.59}},"usage":{"input_tokens":272,"output_tokens":20}}
HTTP 200 | 1.78s（含 DNS；固定 Cloudflare IP 复测 noul=0.60，1.08s）
```

## 证据 2：三问完整判断（tests/test_jev_smoke.py，integration 标记）

命令：`TYPESAFE_API_KEY=... python -m pytest tests/test_jev_smoke.py -m integration -q -s`

输入：demo 青铜器场景 t=2.5s，CENTER_X 偏差 0.08（容差 0.05，即 1.6 倍超限），trigger=OUT_OF_TOLERANCE，
confidence=88。state 为数字+容差，questions=action(choice)/conforms(noul)/quality(score)，criteria 英文。

```
attempt 1: OK in 1.01s
{"action": "PAUSE", "confidence": 0.88, "reason": "jev_choice_pause",
 "backend": "typesafe_jev", "is_reviewed": true, "quality_score": 0.555}
```

判读：对 1.6 倍容差超限，Jev 给出高置信 PAUSE 提案——与本地确定性 Feedback 的 ADJUST/PAUSE
边界一致，属于合理的加码场景（Supervisor 规则：本地 PAUSE 终局，Judge 只能加码）。

## 网络与延迟（中国大陆，无代理）

- 域名解析偶发失败（DNS 抖动）；TLS 握手偶发超过 1.2 秒。整请求往返实测 1.0~1.8 秒。
- 结论：**控制环内抽查保持 Supervisor 默认 1.2 秒超时**（超时按本地决策继续并标"未经复核"）；
  smoke/验证类调用放宽到 10 秒并重试（`TYPESAFE_SMOKE_TIMEOUT`）。
- 演示现场需用现场网络复测；若直连不稳，备选=香港轻量云中转或 OpenRouter。

## 教程核实的关键运营事实（来源：卖家调用教程 docs.qq.com/doc/DQnFjQVpmWmdiWGtC）

- 当前 Key 为**预付额度制：额度用完即止，不能充值续费**；一 Key 一账号，勿公开勿转发。
- 官方**没有余额查询 API**：只能登控制台看 Billing，或自行累加每次响应的 `usage.input_tokens`。
- 计价：输入 $0.042/M tokens，输出免费；单请求上限 64k tokens（state ≤ 32k）；
  限速 250,000 tokens/秒、1200 请求/分钟；429=限速、529=过载、401=Key 无效、422=请求体格式。
- 中文建议（与我们实现一致）：**state 保持中文原文，instructions/criteria 用英文**。
- 账号门户（余额/接码登录，非调用端点）：`https://jev-query.longjinapi.com/`。

## 证据 3：DeepSeek JSON 备用后端（tests/test_judge_deepseek_smoke.py，integration 标记）

命令：`OPENAI_API_KEY=... python -m pytest tests/test_judge_deepseek_smoke.py -m integration -q -s`

```
attempt 1: OK in 1.50s
{"action": "WARN", "confidence": 0.9, "reason": "center_x_deviation_near_tolerance_edge",
 "backend": "openai_json", "is_reviewed": true, "quality_score": 0.75}
```

同一 demo 偏差（CENTER_X 超容差 1.6 倍）下两个后端的对比：

| 后端 | 判断 | 置信度 | quality | 耗时 |
| --- | --- | --- | --- | --- |
| typesafe_jev (jev-1.13.0) | PAUSE | 0.88 | 0.555 | 1.01s |
| openai_json (deepseek-flash) | WARN | 0.90 | 0.75 | 1.50s |

判读：两者均合法（加码方向一致，松紧不同）；`backend` 字段保证复盘可区分。
降级链双后端均真实验证 PASS：Jev 官方主通道 + DeepSeek 国内直连备胎。

## 证据 4：真双后端并行委员会（tests/test_judge_committee_smoke.py，integration 标记）

同一 demo 偏差，Jev 与 DeepSeek **并行**判断后由确定性规则取严融合（`fuse_verdicts`，
纯代码非第三个模型）：

```
attempt 1: OK in 1.75s (parallel wall time)
{"action": "PAUSE", "confidence": 0.89, "reason": "committee_pause_warn",
 "backend": "committee", "is_reviewed": true, "quality_score": 0.63,
 "members": [
   {"action": "PAUSE", "confidence": 0.89, "backend": "typesafe_jev", "quality_score": 0.54},
   {"action": "WARN",  "confidence": 0.88, "backend": "openai_json",  "quality_score": 0.72}]}
```

- 墙钟 1.75s = 较慢成员（DeepSeek 1.5s）的时间，融合零串行开销；串行需 2.5s 以上。
- 融合规则：低置信成员丢弃（阈值默认 0.5）→ 幸存者取最严档 → 置信度取同档最强 →
  quality 取平均 → 成员原始判断全量存档（members 字段）。
- 成员失败/超时自动降级为幸存者决定；全员失败抛 JUDGE_MEMBERS_FAILED → Supervisor
  按本地决策继续并标"未经复核"。
- 接线注意：Supervisor 的 call_timeout_seconds 须大于委员会 member_timeout_seconds
  （建议 +0.3s 余量）。

## 成本核算（本场景）

ping 请求 input_tokens=272（含固定编码开销）；一次完整三问判断约 500~800 input tokens。
按每 0.5 秒抽查 + 事件触发计，一条 8 秒镜头约 25 次调用 ≈ 0.015k tokens × $0.042/M ≈
**每条镜头约 $0.0006**。剩余额度需在控制台人工查看。
