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

## 命令

```cmd
dm-agent --enable-lsp-impact --enable-evidence-graph "修复当前仓库中的问题"
dm-agent-lsp-impact dm_agent/core/agent.py --workspace .
```

第二条命令仅用于手工查看当前文件的报告；要获得“修改前/后”差异，应由 Agent 生命周期
能力在一次写入事务前后自动采样。

外部 MCP 客户端可将 `dm-agent-lsp-impact-mcp` 配置为 stdio 命令，工作目录设为目标仓库。
它暴露与内置 Agent 一致的 `analyze_lsp_impact` 和 `lsp_query` 两个工具 Schema。
