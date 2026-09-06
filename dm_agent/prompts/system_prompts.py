"""系统提示词定义"""

from dm_agent.tools.base import Tool

from .code_agent_prompt import NATIVE_TOOL_SYSTEM_PROMPT, SYSTEM_PROMPT


def build_code_agent_prompt(tools: list[Tool], *, native_tool_calling: bool = False) -> str:
    """从 markdown 文件构建 Code Agent 的系统提示词

    Args:
        tools: 可用工具列表

    Returns:
        系统提示词字符串
    """

    if native_tool_calling:
        return NATIVE_TOOL_SYSTEM_PROMPT

    # 构建工具列表
    tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)

    # 替换模板中的工具占位符
    return SYSTEM_PROMPT.replace("{tools}", tool_lines)
