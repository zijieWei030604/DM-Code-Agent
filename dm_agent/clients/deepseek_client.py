"""DeepSeek Responses/Chat API 的 HTTP 客户端。"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

import requests

from .base_client import BaseLLMClient, LLMError

DEFAULT_RETRY_STATUS_CODES = frozenset({400, 408, 409, 429, 500, 502, 503, 504})


class DeepSeekError(LLMError):
    """当 DeepSeek API 请求失败时抛出。"""


class DeepSeekClient(BaseLLMClient):
    """DeepSeek 聊天补全 API 的轻量级封装。"""

    supports_tool_calling = True

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        endpoint: str = "/v1/chat/completions",
        timeout: int = 600,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        retry_status_codes: Iterable[int] | None = None,
    ) -> None:
        super().__init__(
            api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            # DeepSeek 自带按状态码的内部重试循环；关闭基类重试避免双重退避。
            respond_retries=0,
        )
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be >= 0.")
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_status_codes = (
            DEFAULT_RETRY_STATUS_CODES
            if retry_status_codes is None
            else frozenset(retry_status_codes)
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """向 DeepSeek API 发送聊天式补全请求。"""

        if stream:
            raise NotImplementedError("此客户端未实现流式传输。")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tool_definitions:
            payload["tools"] = [
                {"type": "function", "function": {**definition, "strict": False}}
                for definition in tool_definitions
            ]
        payload.update(extra)

        url = f"{self.base_url}/{self.endpoint.lstrip('/')}"
        for retry_index in range(self.max_retries + 1):
            attempt = retry_index + 1
            has_retry_budget = retry_index < self.max_retries
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if self._is_retryable_exception(exc) and has_retry_budget:
                    self._sleep_before_retry(retry_index)
                    continue
                message = "DeepSeek API request failed"
                if self._is_retryable_exception(exc) and attempt > 1:
                    message = f"{message} after {attempt} attempts"
                raise DeepSeekError(f"{message}: {exc}") from exc

            if response.ok:
                try:
                    return response.json()
                except ValueError as exc:
                    if has_retry_budget:
                        self._sleep_before_retry(retry_index)
                        continue
                    raise DeepSeekError(
                        f"DeepSeek API returned invalid JSON after {attempt} attempts: {exc}"
                    ) from exc

            message = self._format_error(response)
            if self._is_retryable_response(response) and has_retry_budget:
                self._sleep_before_retry(retry_index)
                continue
            if self._is_retryable_response(response) and attempt > 1:
                message = f"{message} after {attempt} attempts"
            raise DeepSeekError(message)

        raise DeepSeekError("DeepSeek API request failed after exhausting retry budget.")

    def extract_text(self, data: dict[str, Any]) -> str:
        """从各种响应格式中提取助手文本内容。"""

        if not isinstance(data, dict):
            raise DeepSeekError("意外的响应负载类型。")

        # Responses API 风格
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        # Chat completions 风格
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        self.last_response_mode = "native_tool_call"
                        self.last_tool_call_count = len(tool_calls)
                        function = tool_calls[0].get("function")
                        self.last_selected_tool = (
                            str(function.get("name", "")) if isinstance(function, dict) else ""
                        )
                        return self._tool_call_as_agent_json(tool_calls[0])
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        self.last_response_mode = "json_fallback"
                        self.last_tool_call_count = 0
                        self.last_selected_tool = ""
                        return content.strip()
                    if isinstance(content, list):
                        parts = [
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "output_text"
                        ]
                        if parts:
                            self.last_response_mode = "json_fallback"
                            self.last_tool_call_count = 0
                            self.last_selected_tool = ""
                            return "".join(parts).strip()

        raise DeepSeekError("无法从 DeepSeek 响应中提取文本。")

    @staticmethod
    def _tool_call_as_agent_json(tool_call: Any) -> str:
        if not isinstance(tool_call, dict):
            raise DeepSeekError("DeepSeek 返回了无效的工具调用。")
        function = tool_call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise DeepSeekError("DeepSeek 工具调用缺少函数名称。")
        raw_arguments = function.get("arguments", "{}")
        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
        except json.JSONDecodeError as exc:
            raise DeepSeekError(f"DeepSeek 工具参数不是有效 JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise DeepSeekError("DeepSeek 工具参数必须是 JSON object。")
        return json.dumps(
            {"thought": "", "action": function["name"], "action_input": arguments},
            ensure_ascii=False,
        )

    @staticmethod
    def _format_error(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        message = f"DeepSeek API error: {response.status_code} {response.reason}"
        if isinstance(body, dict):
            detail = body.get("error", {}).get("message") or body.get("error_msg")
            if not detail:
                detail = body.get("message")
            if detail:
                message = f"{message} - {detail}"
        elif body:
            message = f"{message} - {body}"
        return message

    def _sleep_before_retry(self, retry_index: int) -> None:
        if self.retry_backoff <= 0:
            return
        time.sleep(self.retry_backoff * (2**retry_index))

    def _is_retryable_response(self, response: requests.Response) -> bool:
        return response.status_code in self.retry_status_codes

    @staticmethod
    def _is_retryable_exception(exc: requests.RequestException) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        response = getattr(exc, "response", None)
        return response is None
