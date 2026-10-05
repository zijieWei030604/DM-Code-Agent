# LSP 影响分析

`--enable-lsp-impact` 为本次 Agent run 安装一个独立的 LSP 影响分析能力。它默认调用
`pyright-langserver --stdio`，不自动安装任何语言服务器；可用
`--lsp-impact-command basedpyright-langserver` 覆盖命令。

它与旧的 `--enable-semantic-workspace` 完全独立，不读取或更新 AST / 导入图索引。

## 运行行为

1. 在 `edit_file`、`create_file` 或 `edit_python_symbol` 修改 `.py` 文件前保存该文件的
   内容、符号和诊断快照。
2. 有效修改后通过 LSP 请求 `documentSymbol` 与 `references`，在模型观察中追加最多 10 个
   候选影响文件；单次分析最多记录 200 个引用，超出会显式标记截断。
3. 分析报告写在用户缓存目录 `LOCALAPPDATA/dm-code-agent/lsp-impact/<workspace-hash>/reports/`
   而不是目标仓库。Trace 仅保存报告摘要和 ID。
4. 将报告以 `observation(kind=lsp_impact)` 写入证据图，并以
   `change --derived_from--> observation` 关联当前改动。完成结论会以现有
   `checked_at_completion` 边引用该观察。
5. 修改前基线中不存在、修改后新增的 Error 级 LSP 诊断会阻止完成；服务缺失、超时、影响
   候选尚未验证只会记录为警告或不可用事实。
6. 历史 ImpactReport 始终保留以支持复盘；完成时每个文件只采用最新一份且 `after_hash`
   与当前文件文本哈希一致的报告。后续编辑会使旧报告过期，超时或不可用报告属于尚未验证，
   不会被当作通过，也不会让已修复的历史 Error 持续阻断完成。

## 命令

```cmd
dm-agent --enable-lsp-impact --enable-evidence-graph "修复当前仓库中的问题"
dm-agent-lsp-impact dm_agent/core/agent.py --workspace .
```

第二条命令仅用于手工查看当前文件的报告；要获得“修改前/后”差异，应由 Agent 生命周期
能力在一次写入事务前后自动采样。

外部 MCP 客户端可将 `dm-agent-lsp-impact-mcp` 配置为 stdio 命令，工作目录设为目标仓库。
它暴露与内置 Agent 一致的 `analyze_lsp_impact` 和 `lsp_query` 两个工具 Schema。

## `lsp_query` 的引用结果

`lsp_query(path, action="references")` 在首次查询为空、或只得到当前位置的声明时，会每隔
250ms 最多重试两次，减少语言服务器尚未完成索引造成的空结果。展示给模型的结果最多 200 条：

- 前 50 条附带命中行前后各两行的源码上下文；
- 其余可见条目仅保留 `uri` 与 `range`；
- 返回 `total`、`contextual_items`、`location_only_items` 与 `truncated`，模型可据此决定是否
  再读取某个候选文件。

源码上下文只会读取当前工作区内的文件；工作区外的库或标准库位置始终只返回位置。
