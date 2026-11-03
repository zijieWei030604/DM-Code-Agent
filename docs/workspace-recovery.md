# 工作区快照与 Checkpoint 联合恢复

这是显式开启的恢复能力。它把完整工具批次边界上的工作区内容与 Agent
Checkpoint 绑定；默认恢复到**新目录**，保留原目录、原日志和全部历史快照。
原有 `--checkpoint` 用法不变；未开启本功能时仍只恢复 Agent 状态。

## 使用（Windows CMD）

先在待维护仓库的根目录运行。快照、Checkpoint、trace、报告必须放在工作区外；
Checkpoint 输出使用新文件，不能覆盖原恢复记录。以下盘符与目录可替换。
快照包含源文件和完整 Checkpoint，应使用私有目录保存，不自动加密，也不要提交到公共仓库。

```cmd
dm-agent "完成本次代码维护任务" --workspace-recovery-store E:\agent-recovery\store --checkpoint E:\agent-recovery\run-001.jsonl
python -m dm_agent.recovery list --store E:\agent-recovery\store
python -m dm_agent.recovery restore --store E:\agent-recovery\store --point 上一步显示的point --target E:\AgentProjects\recovered-task
```

恢复命令输出 `Resume checkpoint` 路径。进入新目录，使用已安装的 Agent 环境：

```cmd
cd /d E:\AgentProjects\recovered-task
dm-agent --resume E:\agent-recovery\store\restores\返回的receipt\checkpoint.json --checkpoint E:\agent-recovery\run-002.jsonl
```

也可在原目录使用 `--resume`，但文件必须与该点完全一致；否则拒绝恢复，提示新建
目录。中途 JSONL 中的审计 checkpoint 可能不是可执行恢复点，此时用 `list` 选择已
提交的 point。恢复不会自动运行模型、工具或旧的 pending calls。

## 保存顺序与范围

1. 在首次模型调用前以及完整工具批次边界扫描工作区。
2. 普通文件按 SHA-256 保存为不可变内容对象；相同内容复用对象。
3. 保存清单，包含相对路径、内容哈希、大小、文件权限、目录及 Git base commit。
4. 保存绑定清单的 Checkpoint，最后写 `COMMITTED.json`。没有完整提交标记的点不可恢复。

临时文件写入后 flush/fsync，再原子替换；POSIX 额外 fsync 目录。Windows 不宣称
可抵御所有断电/磁盘控制器故障，恢复时仍逐一校验内容。扫描前后对照可发现捕获期间
的变化；这不是文件系统原子快照，使用前提是没有其他工作区写入者。

Git 工作区包含 tracked 和非 ignored 的 untracked 文件；非 Git 工作区包含范围内
全部普通文件。支持二进制、增删改和空目录。默认不保存 `.env`、依赖目录、缓存与
日志目录。范围内的软链接/junction、子模块、特殊文件拒绝保存，不能静默截断。
默认单文件上限 32 MiB，总内容 512 MiB；超过即停止，不能当作成功快照。

通过 `--workspace-recovery-policy E:\agent-recovery\policy.json` 显式调整：

```json
{
  "include": ["fixtures/ignored.bin"],
  "exclude": ["bench_reports", "swebench_work"],
  "max_file_bytes": 33554432,
  "max_snapshot_bytes": 536870912
}
```

include 可覆盖 Git ignore；不能覆盖 `.git`、依赖等硬排除目录。没有自动 GC，
原始内容不会因压缩或新快照被删除。目录本身的 ACL、所有者、硬链接关系及扩展属性
不属于恢复范围；保存的是普通文件内容与 mode，Windows 只支持其可表达的权限。

## 恢复与验证

Git 使用原仓库仍存在的 base commit 建立 detached worktree，再写入清单内容；
非 Git 使用新目录。逐文件校验后才生成 `READY.json` 和可恢复 Checkpoint。
失败的目标目录保留用于排查，没有 READY 就不能续跑。Git 对象库、源调用日志及
Checkpoint 引用的 LCM 数据库仍须保留；本功能不是可独立迁移的仓库/环境备份。

只重映射已知结构化路径字段，不改写历史对话和任意 Shell 文本。新的工具调用如
显式引用旧工作区会被拒绝，要求重新规划。恢复后清空版本相关 capability 状态，
重新启动 LSP；原证据保留在源 Checkpoint，不能充当当前验证通过。
相对任务初始快照存在改动时，没有新验证首次提醒，之后可按存在验证限制交付；
只读恢复不因缺少测试暂停，现有明确错误拦截策略仍有效。
新验证不等于完整覆盖率或所有业务均正确。

依赖环境与凭据不复制。明确选择现有 Python/Node 环境或自行准备新环境，避免误用
原目录绝对路径。本机制不是沙箱，任意 Shell 仍可能操作工作区外的系统或远程服务。

## 副作用对账

调用日志保留父运行链：未完成的写调用、或恢复点之后的写调用都会阻止盲目恢复。
即使工作区可以还原，也不能推断数据库、消息或 MCP 写操作已经撤回。
操作者检查后显式确认：

```cmd
python -m dm_agent.recovery reconcile --journal E:\agent-recovery\run-001.jsonl.calls.jsonl --call 调用ID --outcome verified --note "已人工核对该调用的实际结果"
```

交互输入 `CONFIRM`。可选择 verified、compensated、not_executed；对账记录追加到
单独日志，并绑定被确认记录的指纹。后续调用或记录变化需要重新核对，原日志不修改。
该命令不是 Agent 工具，但不是针对任意 Shell 的权限隔离措施。不保证外部操作恰好
执行一次。缺失日志、损坏对象、校验失败时拒绝恢复，不自动猜测或重放。

## 离线验证

```cmd
python -m pytest tests\test_workspace_recovery.py tests\test_tool_batches.py -v
```

覆盖新目录还原、内容去重、Git 未提交修改、范围排除、超限、损坏、提交/恢复中断、
跨分叉副作用对账，以及真实 ReAct 工具批次中断不提交半批次状态。

### 本次验证记录（2026-10-19）

实现前本地保存点：`1ee709b`，未推送。以下均为离线检查，没有运行真实模型评测。

| 检查 | 结果 |
| --- | --- |
| 恢复 + tool batches + checkpoint | 51 passed，1 skipped；Windows 无软链接创建权限 |
| 全量 pytest（无本地 MCP 配置的临时 cwd） | 4 项 LSP 缓存权限失败，其余执行项通过；期间新增的恢复边界用例另在上述定向回归通过 |
| LSP 模块复验（独立 LOCALAPPDATA） | 15 passed |
| compileall | 通过 |
| direct_finish/full（隔离 USERPROFILE） | 1/1 通过；原用户目录运行因 SQLite 访问权限失败 |
| maintenance --list | 通过 |
| 本次变更 Ruff / Black | 通过 |
| 全仓 Ruff / Black / mypy | 原有 1 条 lint、21 个格式文件、18 条类型错误；未混入无关修复 |

本次没有宣称磁盘断电、远程副作用恰好一次、任意 Shell 沙箱或真实模型任务通过率得到验证。
