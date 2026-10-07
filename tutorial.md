# TraceCode 简历源码课程

源码范围：2026-10-05 当前工作区；按简历分点逐课讲解，不将既往评测数字视为本轮验证结果。

1. Agent Runtime 与扩展机制（已展开，见下文）
2. 分层压缩与历史检索（待展开）
3. 执行审计与恢复（待展开）
4. 影响分析与证据验收（待展开）
5. Benchmark 与评测口径（待展开）

## 1. Agent Runtime 与扩展机制

场景：用户要求调查并修复代码，模型选择工具，宿主执行工具，再将结果交回模型。

阅读主链：
`dm_agent/cli/runner.py:create_agent` → `dm_agent/core/agent.py:ReactAgent.run/_run_once`
→ `dm_agent/core/tool_invoker.py:ToolInvoker.invoke` → `dm_agent/core/events.py:EventBus`
→ `dm_agent/core/completion.py:CompletionGate.review`。

### 装配与执行

CLI 根据配置组装工具、SkillManager 和 capabilities，交给 ReactAgent。
工具统一表示为 Tool（name、description、runner、input_schema、read_only 等）。
每步构建上下文和工具定义，调用模型，解析 action/action_input，查找工具并执行。
工具反馈成为 observation，进入后续上下文。完成候选经过 before_finish，不由模型单独决定。

### Function Calling

Tool.function_definition 提供供应商中立的函数定义。clients/openai_client.py 和
clients/deepseek_client.py 适配原生函数调用，转换成内部 action/action_input。
当前单步内核只消费一个动作；模型一次返回多个工具调用不代表主循环并行执行。
工具真实执行发生在本地 runner；Schema 描述不等于工具语义正确或完整安全校验。

### 工具执行

ToolInvoker 先检查参数是对象，再发布 before_tool_call，允许拦截或修改参数。
写入前按现有策略备份，执行 result_runner 或 execute，捕获异常，统一结果展示并限制观察长度，
构建执行事实，发布 after_tool_result。ToolResult.status 优先用于成功判断；字符串结果存在旧兼容启发式。

### 生命周期

六个事件：on_run_start、before_llm_request、before_tool_call、after_tool_result、before_finish、on_run_end。
Capability.install 使用 context.event_bus.on 注册处理器；普通处理器依注册顺序运行。
observer 收到事件深拷贝，其返回值不参与决策；policy 用于 before_tool_call/before_finish，
这两个阶段遇到 block 后停止该前置链。故“六个可拦截钩子”更准确可表述为“六个生命周期钩子，支持工具与完成拦截”。

### Skills

skills/manager.py 发现内置、用户、项目技能；skills/runtime.py:prepare_skills 将名称、描述及推荐标记
加入提示词，并注册 load_skill/load_skill_resource。模型按需调用工具获取正文和资源。
首次激活会检查工具名冲突、注册附加工具，记录内容哈希与来源。重复读取不会重复激活，仍可能重复产生正文观察。

### MCP

mcp/manager.py 将远程工具定义转换为普通 Tool，runner 转发 tools/call。
变更通知触发 refresh_server_tools，重新 tools/list，再通知 Agent.refresh_mcp_tools 更新目录。
此能力依赖服务端通知支持，不代表本地配置文件监控或热重载服务端程序。

### 前台只读子 Agent

subagents/manager.py 用线程池调度独立 Python 子进程，实际子运行具有自己的模型客户端、ReAct 和历史。
同批任务独立，主循环等待整批；任务依赖必须分批。工具白名单限制读取能力，不是操作系统沙箱。
宿主取消或超时触发进程清理；task 返回整体结果时仍需检查每项状态，整体 success 不等于每项成功。
worker.py 的 submit_exploration_result 提交结构化报告，校验格式不等于验证结论。

### 复述练习

以“先调查配置入口和测试，再修复”为例说明：谁选择工具、谁执行工具、何处并行、何处拦截、
为什么子 Agent 报告通过 Schema 校验仍不能证明代码正确。
