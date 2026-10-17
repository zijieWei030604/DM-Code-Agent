# DeepSeek 流式工具调用

`deepseek-chat` 的普通 Agent 请求可以使用 OpenAI 兼容 SSE 流。流式能力只改变模型
输出的接收方式，不改变 ReAct 的一轮一工具执行语义。

## 状态与执行边界

每次请求会产生 `stream_start`、文本/思考增量、`toolcall_start`、`toolcall_delta`、
`toolcall_end`，最后以 `stream_done` 或 `stream_error` 结束。`StreamToolCall` 按调用
索引累积参数 JSON；只有收到终止事件并且完整 JSON 对象解析成功后，才构造最终响应。

因此，参数片段、部分 JSON 与 UI 回调都**不能**触发工具。完成的调用仍然经过既有
`ToolInvoker` 链路：参数校验、`before_tool_call`、备份、工具执行、输出截断及
`after_tool_result`。全量原始 LLM I/O 的持久化策略保持不变；默认 trace 仅记录流的
边界事件和元数据，不记录文本增量。

当前 Runtime 每个 ReAct step 只允许一个原生工具调用。DeepSeek 返回多个调用时，客户端
会显式返回协议错误，由既有原生工具调用修复重试要求模型仅调用一个；不会执行或静默丢弃
其中任一调用。

## 提前阻断

在 `toolcall_delta` 时，解析器只暴露已经闭合的顶层标量字段。例如完整的 `path` 已出现
且解析到工作区外路径时，Runtime 可以终止本次请求，并记录 `stream_security_blocked`。
这只是确定性的提前拒绝：最终参数校验及全部生命周期策略仍在 `toolcall_end` 后执行。

该实现参考了 Oh My Pi 的事件分层、分片参数累积与终止事件语义；代码为本项目的 Python
实现，沿用本项目的单工具 ReAct 协议、追踪和安全边界。
