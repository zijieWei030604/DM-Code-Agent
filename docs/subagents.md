# 前台只读子 Agent

此功能复用 ReAct，通过 CLI 外层装配注册 `task`、`task_list`、`task_result`。
不新增内核功能分支，不包含后台 Agent / A2A / 多人写入。

## 启用（Windows CMD）

```cmd
uv run --frozen dm-agent --subagent-store E:\AgentRuns\explore-session-01 --subagent-workers 3 --subagent-timeout 180 --enable-evidence-graph "先定位配置加载入口，再并行调查实现和测试，汇总依据后再修改。"
```

`--subagent-store` 同时显式启用私有持久化：该目录保存子任务说明、结果、工具来源、
完整会话 Checkpoint。不要把它作为可公开 trace 分享。模型原始请求/响应的 trace 捕获
仍然关闭。密钥通过子进程 stdin 传递，不写入任务记录或命令行。目录必须位于工作区外，
防止污染 changed-files 评分；同一目录绑定一个工作区，同一时刻仅允许一个管理器。
新任务建议新目录；恢复原会话时传回同一个目录。不开启此参数不改变原有运行。

## 模型接口

```json
{
  "context": "已定位 pricing.py:calculate_total，调查舍入问题，不修改代码。",
  "tasks": [
    {"role": "explore", "instruction": "分析根因，给出文件与行号依据。"},
    {"role": "explore", "instruction": "调查相关测试的边界条件覆盖。"}
  ]
}
```

每批 1–16 项；同批任务必须独立。有依赖时主 Agent 先提交定位批次，返回后再提交调查批次。
每项是独立 Python 进程、模型客户端、ReAct 实例、历史、工具集合、trace 和 checkpoint。
线程池仅在父进程调度/等待这些进程。并发上限 1–8，同一个管理器的批次串行进入，所以
不会因重复调用绕过上限。父工具等待所有进程实际退出，才返回逐项状态。

返回的 `id` 是本次 attempt，`session_id` 是逻辑子会话。追问时在任务项传入
`session_id`（也接受该会话的任意 attempt ID），使用最新 attempt 已保存的历史开启新一轮。
若最近 attempt 尚未保存 Checkpoint，则沿历史回退到最近已有 Checkpoint 的 attempt。
模型须重新读取可能因父 Agent 修改而过期的代码。不会继续使用旧步骤计数冒充原请求仍在运行。

`task` 返回每项的状态、简短语义 `summary` 与 `output`：完整输出不超过 5,000 字符时直接
放入 `output`；超过时返回按行截断的预览，并标记 `output_truncated=true`、
`output_char_count`。此时再调用 `task_result` 展开完整内容。`task_list({"offset":0})`
分页列出同样的轻量结果。`task_result` 接收 `task_id`、`section`（answer/evidence）、`offset`
和 `limit`，按字符分页展开完整答案或 Runtime 记录的工具来源。模型引用的文件行号是声明；
真实工具调用、参数、步骤和运行 ID 单独记录。

## 结构化输出

默认要求 `summary: string, findings: string[], uncertainties: string[]`。Explore 完成时调用
`submit_exploration_result({"report": ...})`；原生 Function Calling 模型将 `report` 作为函数参数
提交，Runtime 直接校验并落盘，不再从最终回答文本提取报告。对于不发出工具调用的兼容模型，
仍支持解析旧 `finish` JSON 报告，并标注为 `legacy_finish_json`，方便在 Trace 中区分。
可按项提供 `output_schema`。实现明确的 JSON Schema 子集：
`type`（object/array/string/boolean/integer/number/null）、`properties`、`required`、
`additionalProperties`（仅布尔）、`items`、`enum`、`description`，嵌套最多十层。
不支持 `$ref`、组合 Schema 等关键字；在启动任何子进程前拒绝不支持的契约，不静默忽略。
格式不符时将具体校验错误反馈模型；首次无效交付后最多允许两次纠正（合计三次无效
交付即耗尽），完成工具和旧 `finish` 共用预算。纠正沿用当前 ReAct 循环，不重置步数、
Token 或墙钟预算；耗尽后不再向模型发送请求。不增加 `partial` 状态，也不绕过 Schema。
只有正常完成并通过同一 Schema 校验才能成功；模型错误或步数耗尽不能被 JSON 兜底洗成成功。
失败时 `report=null`、`schema_valid=false`，结果中包含 `validation_errors`、
`rejected_delivery_count` 和 `rejections_ref`。被拒绝的候选原文及原因追加到私有尝试目录的
`rejected_submissions.jsonl`，不塞回父模型摘要，也不删除原 Trace / Checkpoint。
`schema_valid=true` 仅说明格式合格，不表示调查结论已被验证。主 Agent 可以显式追问。

## 权限与一致性

Explore 只注册经过选择的内置目录、搜索、读取工具；不继承项目扩展、MCP、可执行 Skills、
Shell、Python 执行或委派工具。路径规范化限制在工作区内；扫描遇到指向工作区外的链接拒绝。
开启父 LSP 功能时，子进程创建自己的 LSP 服务，只开放 symbols/references/definition/diagnostics。
这是一套工具权限边界，不是操作系统沙箱。会话记录写入私有目录，业务工作区只读。

前台父循环在任务工具中等待，不会同时编辑。不防范外部编辑器，符合当前使用前提。
父 Agent 修改后仍使用现有证据版本规则；不把调查意见作为 verification。
证据图复用 observation 节点与现有计划关联，metadata 包含子任务状态和 trace 引用。

## 状态、取消和恢复

状态：queued / running / succeeded / failed / cancelled / timed_out / interrupted。
执行状态和 delivery 分开；returned 仅表示结果组装并交回工具调用，不等于模型采纳。
一个失败不会取消同批其他独立任务。`TaskManager.cancel()` 是宿主取消入口；CLI 的
Ctrl+C 会取消整批、终止并回收所属子进程树。前台运行期间模型无法再调用取消工具，
因此不提供一个实际上无法并发调用的 `task_cancel` 模型接口。

每项最多 min(父 max_steps, 30) 步，默认 180 秒墙钟预算（不含排队），请求超时 30 秒；
累计模型文本估算预算 32000 Token，下一次请求前检查，正在生成的单次响应可能越过软额度。
到墙钟预算即终止进程树，并等待退出后返回。父进程异常退出时 stdin 关闭，工作进程的
owner watcher 终止自身进程树，不作为后台守护服务继续运行。

任务 journal 是 append-only，恢复不覆盖原状态；最后一条残缺记录用追加恢复标记保留。
重新打开目录时未完成项标记 interrupted，不自动重跑；主 Agent 通过 task_list 查询、
task_result 复用完成结果，或明确委派新 attempt。追问恢复最后一个有效 checkpoint 的历史，
不承诺恢复正在进行的 HTTP 请求。不同 attempt 使用不同目录，原始记录保留。

## 指标与测试

父结果 metadata.subagents 含子运行指标，subagent_estimated_tokens 是子运行文本估算总和；
不等于供应商账单 Token，也不自动计入既有 Benchmark 的父客户端计数。A/B 分析必须另外加上
子用量，并区分 schema/工具定义、重试等估算未覆盖部分。被强制终止的请求可能没有完整用量，
subagent_usage_complete 为 false，不把未知输出记作确定的零消耗。

```cmd
uv run --frozen pytest tests/test_subagents.py -v
```

测试使用脚本模型和本机假工作进程，不需要 API Key。真实模型收益需要后续运行同题对照；
本实现不声明通过率、Token 节省或提速指标。参考架构而非复制上游 TypeScript：
[Oh My Pi task](https://github.com/can1357/oh-my-pi/tree/main/packages/coding-agent/src/task)。

### 本次离线验证（2026-10-03）

- 新增 19 项委派测试全部通过：实际进程并行与并发上限、部分失败、超时、取消、
  父进程失联收尾、恢复查询、Checkpoint 追问、权限限制、原生/提示词 JSON 两种结构化
  完成通道、输出契约和证据图关联。
- compileall 与本次修改文件的 Ruff / Black 检查通过；维护 Benchmark 清单可加载。
- 完整 pytest 出现 5 项失败：3 项原有工具清单断言仍按旧 LSP 工具数量计算，
  1 项 LSP 报告目录写权限失败，1 项嵌套 pytest 跨目录收集被本机权限拒绝。
  前 4 项已在 HEAD 的独立副本中复现，第 5 项单独运行仍因目录权限失败。
- direct_finish 的原始与当前版本均因 SQLite 文件无法打开而未成功；不能用退出码 0
  代替评测成功。全库 Ruff、Black、mypy 也仍有未修改文件中的存量问题。
- 未运行真实模型或收费评测，不将本次功能测试解释为实际任务成功率提升。
