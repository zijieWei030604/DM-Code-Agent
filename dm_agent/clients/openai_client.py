"""OpenAI API 的客户端（使用官方 SDK）。"""

from __future__ import annotations

import json
from typing import Any

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_client import BaseLLMClient, LLMError, classify_retryable_exception


class OpenAIClient(BaseLLMClient):
    """OpenAI API 的轻量级封装（使用官方 SDK）。"""

    supports_tool_calling = True
    supports_json_schema = True

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-5",
        base_url: str = "",
        timeout: int = 600,
        respond_retries: int = 2,
    ) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai 未安装。请运行: pip install openai")

        super().__init__(
            api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            respond_retries=respond_retries,
        )

        # Keep the official default when no endpoint override is configured,
        # while allowing OpenAI-compatible providers to supply their endpoint.
        client_options: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            client_options["base_url"] = self.base_url
            # Compatibility endpoints vary; only advertise strict Responses
            # schemas for the official endpoint unless a provider opts in later.
            self.supports_json_schema = False
        self.client = OpenAI(**client_options)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tool_definitions: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """向 OpenAI API 发送生成请求。"""

        try:
            # 调用 OpenAI responses API
            request: dict[str, Any] = {"model": self.model, "input": messages}
            if tool_definitions:
                request["tools"] = [
                    {"type": "function", **definition, "strict": False}
                    for definition in tool_definitions
                ]
                request["tool_choice"] = extra.get("tool_choice", "auto")
            if json_schema:
                request["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "schema": json_schema,
                        "strict": True,
                    }
                }
            response = self.client.responses.create(**request)

            # 返回包含响应的字典
            return {"response": response}

        except Exception as e:
            raise LLMError(
                f"OpenAI API 调用失败: {e}", retryable=classify_retryable_exception(e)
            ) from e

    def extract_text(self, data: dict[str, Any]) -> str:
        """从 OpenAI 响应中提取文本内容。"""

        if not isinstance(data, dict):
            raise LLMError("意外的响应负载类型。")

        # 从响应中提取文本
        response = data.get("response")
        if response:
            try:
                for item in getattr(response, "output", ()):
                    if getattr(item, "type", "") == "function_call":
                        calls = [
                            candidate
                            for candidate in getattr(response, "output", ())
                            if getattr(candidate, "type", "") == "function_call"
                        ]
                        arguments = json.loads(getattr(item, "arguments", "{}"))
                        if not isinstance(arguments, dict):
                            raise ValueError("tool arguments must be a JSON object")
                        self.last_response_mode = "native_tool_call"
                        self.last_tool_call_count = len(calls)
                        self.last_selected_tool = str(item.name)
                        return json.dumps(
                            {
                                "thought": "",
                                "action": item.name,
                                "action_input": arguments,
                            },
                            ensure_ascii=False,
                        )
                self.last_response_mode = "json_fallback"
                self.last_tool_call_count = 0
                self.last_selected_tool = ""
                return response.output_text.strip()
            except Exception as e:
                raise LLMError(f"无法从 OpenAI 响应中提取文本: {e}") from e

        raise LLMError("无法从 OpenAI 响应中提取文本。")

    def close(self) -> None:
        self.client.close()

    def _convert_messages_to_input(self, messages: list[dict[str, str]]) -> str:
        """将标准消息格式转换为输入字符串。"""
        input_parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                input_parts.append(f"System: {content}")
            elif role == "user":
                input_parts.append(f"User: {content}")
            elif role == "assistant":
                input_parts.append(f"Assistant: {content}")
            else:
                input_parts.append(content)

        return "\n\n".join(input_parts)
