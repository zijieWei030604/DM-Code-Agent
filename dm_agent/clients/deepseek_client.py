"""DeepSeek Responses/Chat API 的 HTTP 客户端。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import requests

from .base_client import BaseLLMClient, LLMError
from .streaming import StreamEvent, StreamToolCall

DEFAULT_RETRY_STATUS_CODES = frozenset({400, 408, 409, 429, 500, 502, 503, 504})


class DeepSeekError(LLMError):
    """当 DeepSeek API 请求失败时抛出。"""


class DeepSeekStreamBlocked(DeepSeekError):
    """A deterministic stream-time policy rejected a completed argument field."""


class DeepSeekClient(BaseLLMClient):
    """DeepSeek 聊天补全 API 的轻量级封装。"""

    supports_tool_calling = True
    supports_json_schema = True
    supports_streaming = True

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
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
        on_stream_event: Callable[[StreamEvent], bool | None] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """向 DeepSeek API 发送聊天式补全请求。"""

        if json_schema is not None:
            if stream:
                raise ValueError("DeepSeek JSON Schema responses do not use this chat stream path.")
            return self._complete_structured_response(messages, json_schema, extra)

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
        if stream:
            return self._post_stream_with_retry(url, payload, on_stream_event=on_stream_event)
        return self._post_with_retry(url, payload)

    def _post_stream_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        on_stream_event: Callable[[StreamEvent], bool | None] | None,
    ) -> dict[str, Any]:
        """Consume one DeepSeek OpenAI-compatible SSE response into a final payload.

        The callback sees immutable events.  Returning ``False`` rejects the
        request immediately; it never authorizes or executes a tool call.
        """

        request_id = uuid.uuid4().hex
        stream_payload = {**payload, "stream": True}
        for retry_index in range(self.max_retries + 1):
            has_retry_budget = retry_index < self.max_retries
            response: Any | None = None
            try:
                response = self.session.post(
                    url, json=stream_payload, timeout=self.timeout, stream=True
                )
                if not response.ok:
                    message = self._format_error(response)
                    if self._is_retryable_response(response) and has_retry_budget:
                        self._sleep_before_retry(retry_index)
                        continue
                    raise DeepSeekError(message)
                return self._consume_sse(response, request_id, on_stream_event)
            except DeepSeekStreamBlocked as exc:
                self._emit_stream_error(on_stream_event, request_id, str(exc))
                raise
            except DeepSeekError as exc:
                self._emit_stream_error(on_stream_event, request_id, str(exc))
                raise
            except requests.RequestException as exc:
                if self._is_retryable_exception(exc) and has_retry_budget:
                    self._sleep_before_retry(retry_index)
                    continue
                error = DeepSeekError(f"DeepSeek stream request failed: {exc}")
                self._emit_stream_error(on_stream_event, request_id, str(error))
                raise error from exc
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        raise DeepSeekError("DeepSeek stream request failed after exhausting retry budget.")

    @staticmethod
    def _emit_stream_event(
        callback: Callable[[StreamEvent], bool | None] | None, event: StreamEvent
    ) -> None:
        if callback is not None and callback(event) is False:
            raise DeepSeekStreamBlocked(event.reason or "Stream-time policy blocked this request.")

    @staticmethod
    def _emit_stream_error(
        callback: Callable[[StreamEvent], bool | None] | None, request_id: str, reason: str
    ) -> None:
        """Best-effort terminal notification; a policy cannot reject an error event."""

        if callback is None:
            return
        try:
            callback(StreamEvent("stream_error", request_id, reason=reason))
        except Exception:
            # A UI observer must not mask the original transport or policy failure.
            return

    def _consume_sse(
        self,
        response: Any,
        request_id: str,
        callback: Callable[[StreamEvent], bool | None] | None,
    ) -> dict[str, Any]:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: dict[int, StreamToolCall] = {}
        saw_done = False
        finish_reason = ""
        self._emit_stream_event(callback, StreamEvent("stream_start", request_id))
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError as exc:
                raise DeepSeekError(f"DeepSeek stream yielded invalid JSON: {exc}") from exc
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                if not text_parts:
                    self._emit_stream_event(callback, StreamEvent("text_start", request_id))
                text_parts.append(text)
                self._emit_stream_event(callback, StreamEvent("text_delta", request_id, delta=text))
            thinking = delta.get("reasoning_content")
            if isinstance(thinking, str) and thinking:
                if not thinking_parts:
                    self._emit_stream_event(callback, StreamEvent("thinking_start", request_id))
                thinking_parts.append(thinking)
                self._emit_stream_event(
                    callback, StreamEvent("thinking_delta", request_id, delta=thinking)
                )
            raw_calls = delta.get("tool_calls")
            if isinstance(raw_calls, list):
                for offset, raw_call in enumerate(raw_calls):
                    if not isinstance(raw_call, dict):
                        continue
                    index = raw_call.get("index", offset)
                    if not isinstance(index, int):
                        continue
                    state = tool_calls.get(index)
                    function = raw_call.get("function")
                    function = function if isinstance(function, dict) else {}
                    if state is None:
                        state = StreamToolCall(index=index)
                        tool_calls[index] = state
                        self._emit_stream_event(
                            callback,
                            StreamEvent(
                                "toolcall_start",
                                request_id,
                                tool_index=index,
                                tool_call_id=str(raw_call.get("id") or ""),
                                tool_name=str(function.get("name") or ""),
                            ),
                        )
                    if raw_call.get("id"):
                        state.call_id = str(raw_call["id"])
                    if function.get("name"):
                        state.name = str(function["name"])
                    arguments = function.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        state.append(arguments)
                        self._emit_stream_event(
                            callback,
                            StreamEvent(
                                "toolcall_delta",
                                request_id,
                                delta=arguments,
                                tool_index=index,
                                tool_call_id=state.call_id,
                                tool_name=state.name,
                                arguments_preview=state.preview(),
                            ),
                        )
            reason = choice.get("finish_reason")
            if isinstance(reason, str) and reason:
                finish_reason = reason
        if not saw_done and not finish_reason:
            raise DeepSeekError("DeepSeek stream ended before a finish reason or [DONE].")
        if text_parts:
            self._emit_stream_event(callback, StreamEvent("text_end", request_id))
        if thinking_parts:
            self._emit_stream_event(callback, StreamEvent("thinking_end", request_id))
        serialized_calls: list[dict[str, Any]] = []
        for index in sorted(tool_calls):
            state = tool_calls[index]
            try:
                state.finalize()
            except ValueError as exc:
                raise DeepSeekError(f"DeepSeek streamed invalid tool arguments: {exc}") from exc
            self._emit_stream_event(
                callback,
                StreamEvent(
                    "toolcall_end",
                    request_id,
                    tool_index=index,
                    tool_call_id=state.call_id,
                    tool_name=state.name,
                    arguments_preview=state.arguments or {},
                ),
            )
            serialized_calls.append(
                {
                    "id": state.call_id or f"stream_call_{index}",
                    "type": "function",
                    "function": {
                        "name": state.name,
                        "arguments": state.raw_arguments,
                    },
                }
            )
        message: dict[str, Any] = {"content": "".join(text_parts) or None}
        if serialized_calls:
            message["tool_calls"] = serialized_calls
        if thinking_parts:
            message["reasoning_content"] = "".join(thinking_parts)
        self._emit_stream_event(
            callback, StreamEvent("stream_done", request_id, reason=finish_reason)
        )
        return {"choices": [{"message": message, "finish_reason": finish_reason}]}

    def complete_summary(
        self, messages: list[dict[str, str]], *, max_tokens: int, timeout: float = 60.0
    ) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/{self.endpoint.lstrip('/')}",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def _complete_structured_response(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Use the Responses API for provider-enforced JSON Schema output.

        DeepSeek exposes strict JSON Schema output through its Responses API.
        Ordinary agent tool calls retain the existing Chat Completions path.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_response",
                    "schema": json_schema,
                    "strict": True,
                }
            },
            **extra,
        }
        return self._post_with_retry(f"{self.base_url}/responses", payload)

    def _post_with_retry(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit one provider payload with the client's existing retry policy."""

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

        # Responses API can return text only inside output message parts.
        output = data.get("output")
        if isinstance(output, list):
            parts = [
                part.get("text", "")
                for item in output
                if isinstance(item, dict)
                for part in item.get("content", [])
                if isinstance(part, dict) and part.get("type") == "output_text"
            ]
            text = "".join(part for part in parts if isinstance(part, str)).strip()
            if text:
                self.last_response_mode = "json_schema"
                self.last_tool_call_count = 0
                self.last_selected_tool = ""
                return text

        # Chat completions 风格
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        if len(tool_calls) != 1:
                            self.last_response_mode = "native_tool_call_multiple"
                            self.last_tool_call_count = len(tool_calls)
                            self.last_selected_tool = ""
                            return (
                                "DeepSeek returned multiple native tool calls in one ReAct step. "
                                "Return exactly one tool call."
                            )
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

    def close(self) -> None:
        self.session.close()

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
            # Treat malformed provider output like any other response parse failure.
            # Returning plain text routes it through ReactAgent's existing recovery
            # loop instead of aborting the entire task as an agent exception.
            return f"DeepSeek tool arguments are not valid JSON: {exc}"
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
