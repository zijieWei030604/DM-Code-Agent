# DM-Code-Agent 项目导学

> 目标：形成可用于面试陈述的项目主线，而不是逐文件背代码。内容基于当前分支 `codex/semantic-workspace-impact`、提交 `95bd6e2` 与仓库内评测产物。

## 1. 前置知识

| 知识点 | 为什么需要 | 项目位置 | 高频度 |
| --- | --- | --- | --- |
| ReAct、Plan-and-Execute | 理解规划与逐步工具执行 | `dm_agent/core/agent.py`、`planner.py` | 高 |
| Function Calling、JSON Schema | 理解工具选择和参数校验 | `clients/`、`tools/base.py`、`core/tool_invoker.py` | 高 |
| 事件总线、插件机制 | 理解六个生命周期钩子 | `core/events.py`、`capabilities.py` | 高 |
| 上下文窗口、检索 | 理解裁剪、原子记忆和召回 | `memory/`、`core/context_window.py` | 高 |
| SQLite、FTS5、BM25、AST | 理解增量语义索引 | `workspace/` | 中高 |
| JSONL、Checkpoint | 理解审计、恢复、回放和分叉 | `tracing/`、`core/checkpoint.py` | 高 |
| Pytest、Docker | 理解两套 Benchmark | `benchmarks/`、`swebench_verified/` | 高 |

## 2. 亮点与学习顺序

| 顺序 | 亮点 | 读完要回答 |
| --- | --- | --- |
| 1 | 单一 ReAct 内核 | 一次 run 如何从任务走到完成？ |
| 2 | 六个可拦截钩子 | 为什么扩展能力不必侵入主循环？ |
| 3 | 结构化工具框架 | 工具成功、失败、不可用由谁判断？ |
| 4 | 确定性上下文压缩 | 旧消息怎样变成原子记忆，怎样防止负收益？ |
| 5 | 语义工作区 | 如何增量维护符号、引用和影响关系？ |
| 6 | 证据图与 Trace | 怎样证明修改有依据且经过验证？ |
| 7 | 评测闭环 | 自建隐藏测试与官方 Harness 分别证明什么？ |

## 3. 必备知识检查表

- [ ] 能画出 `run -> _run_once -> LLM -> ToolInvoker -> observation -> next step`。
- [ ] 能区分 Planner 计划项与 ReAct 最大循环步数。
- [ ] 能解释六个钩子的时机、数据权限和异常隔离。
- [ ] 能解释 `ToolResult.status/error_code/exit_code` 与展示文本的区别。
- [ ] 能说明 read-before-edit、写前备份、观察截断各防什么。
- [ ] 能说明压缩触发、记忆生成、检索、正收益提交与回滚。
- [ ] 能区分会话日志、Trace、Checkpoint、回放和分叉。
- [ ] 能解释语义索引增量更新与影响边局部重建。
- [ ] 能解释证据图何时注入模型、何时阻止完成。
- [ ] 能解释严格通过、隐藏测试、official resolved、F2P、P2P。

## 4. 推荐阅读

| 阶段 | 文件 | 读完应能回答 |
| --- | --- | --- |
| 主循环 | `dm_agent/core/agent.py` | 初始化、规划、循环、完成、重试怎样串联？ |
| 工具 | `core/tool_invoker.py`、`tools/base.py` | Schema、钩子、备份、执行、截断、状态怎样串联？ |
| 扩展 | `core/events.py`、`core/capabilities.py` | 钩子怎样注册、串联、阻断和容错？ |
| 上下文 | `core/context_window.py`、`memory/context_compressor.py` | 压缩候选怎样产生、核算与提交？ |
| 代码理解 | `workspace/engine.py`、`workspace/impact.py` | AST 事实怎样入库，变更后哪些边重建？ |
| 可解释性 | `core/evidence.py`、`extensions/capabilities/evidence.py` | 节点、边、版本与完成门怎样工作？ |
| 持久化 | `tracing/writer.py`、`session.py`、`replay.py`、`fork.py`、`core/checkpoint.py` | 日志怎样形成可导航、可恢复历史？ |
| 自建评测 | `benchmarks/runner.py`、`models.py`、`tasks.py` | 快照、隐藏测试、严格判分怎样实现？ |
| 公开评测 | `swebench_verified/predict.py`、`evaluate.py`、`analyze.py` | 预测与官方 Docker 判分为何分离？ |

看不懂具体文件时，可以继续让 AI 按调用链解释；本文只提供阅读路径、关键问题和验收标准。

## 5. 技术定位

这是面向本地代码维护的轻量 Agent Runtime。主循环只负责请求模型、选择动作、执行工具、回填观察和判断结束；安全、上下文、证据与重规划通过事件和 Capability 装配。工程重点是让长链路执行可约束、可恢复、可审计、可评测。

## 6. 核心原理

### 6.1 单一执行循环

`run()` 管理多次尝试，`_run_once()` 管理一次 ReAct 循环；每轮只处理一个动作，工具结果作为下一条观察进入历史。Planner 提供初始路线，失败后可按配置重规划，但不代替执行循环。状态集中在 `RunContext`，步骤统一为 `Step`，完成动作统一经过 `CompletionGate`。

### 6.2 生命周期扩展

事件总线覆盖运行开始、模型请求前、工具调用前、工具结果后、完成前和运行结束。中间件可改写数据，策略可阻断危险动作或不可信完成，观察者只记录事实。处理器按注册顺序执行；普通处理器异常会回滚局部修改并记录错误，策略不可用时保守阻断。

### 6.3 确定性上下文压缩

满足旧消息数量且达到轮次间隔或 token 预算时，压缩器把较旧消息提炼为本地原子记忆，保留近期消息，再根据最近上下文召回相关记忆，全程不调用额外 LLM。压缩前保存状态，候选结果重新估算 token；只有 `after < before` 才提交，否则恢复记忆、计数器与最近有效折叠。

### 6.4 语义工作区

系统以文件哈希识别变化，用 Python AST 提取符号和引用并写入 SQLite；搜索优先使用 FTS5/BM25，必要时回退普通查询。文件被有效修改后，仅更新相关事实并重算影响；局部更新会清除可能过期 source 的旧出边，再结合最新符号和引用重建。

### 6.5 证据、审计与恢复

证据图连接需求、计划、读取、变更、验证和结论。状态变化时可向模型注入有界摘要；已知失败验证与成功声明矛盾时，完成门最多阻断一次。图变化追加到 Trace。JSONL 条目具有 `id/parent_id`；Checkpoint 保存历史、步骤、计划、metadata、记忆和 Capability 状态。任意条目都可形成分叉，但只有此前存在 checkpoint 时才能直接续跑。

## 7. 关键设计决策

| 决策 | 取舍 | 风险与验证 |
| --- | --- | --- |
| 单循环 + Capability | 主链稳定、功能可拆，但事件契约必须清晰 | 钩子顺序、异常和阻断测试 |
| 结构化 `ToolResult` | 状态优先于自然语言；旧工具保留文本兜底 | 覆盖 success/failed/unavailable |
| 本地确定性压缩 | 无额外模型调用，但语义抽象弱于 LLM 摘要 | 正收益门、完整回滚、专项 A/B |
| 增量 SQLite 索引 | 减少重复解析，但要防陈旧边 | 局部与全量 edge set 等价测试 |
| append-only JSONL | 易审计和分叉，但日志持续增长 | schema 兼容、脱敏、读侧重建 |
| 官方 SWE Harness | 结果口径可比，但环境成本高 | 以 F2P/P2P 和 official resolved 为准 |

## 8. 量化结果与证据边界

| 主张 | 当前证据 | 结论 |
| --- | --- | --- |
| 压缩 12.71% | `bench_reports/context-pressure-4k.json`：44 次有效折叠，`182921 -> 159676` | 可核验；是压缩检查点直接降幅，不是全部请求总 token 降幅 |
| 自建 30 题 | `bench_reports/final-30-scoped.json`：严格与隐藏测试均为 28/30 | 当前可核验值是 93.3%；简历中的 26/30 暂无对应报告 |
| SWE 35/50 | 当前 50 题报告 `swebench-verified-crossrepo-50-20260806.json` 为 21/50 | 35/50 暂不可核验，不能在面试中当作已有仓库证据 |
| 项目测试 | 用户最近提供输出：641 passed、1 skipped | 简历定稿前建议重跑并保留 CI 或终端产物 |

## 9. 面试前待办

1. 保存 `35/50` 对应的官方 Harness JSON、predictions 与运行配置，并确认分母是 submitted 50。
2. 最终确定自建 Benchmark 使用 `28/30` 还是另一次 `26/30`，每个数字对应唯一报告。
3. 保存上下文实验命令、模型、温度、预算与触发范围，避免夸大 12.71% 的统计口径。
4. 准备一个定位失败案例和一个 P2P 回归案例，主动说明系统边界与优化方向。
