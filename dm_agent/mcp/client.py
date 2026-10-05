"""MCP 客户端 - 负责与单个 MCP 服务器通信"""

import contextlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any

from .oauth import OAuthAuthorizer, OAuthError


class MCPClient:
    """
    MCP客户端负责与单个MCP服务器进程进行通信，通过标准输入/输出与外部MCP服务器交互，
    实现工具列表获取和工具调用等功能。该客户端使用多线程处理服务器响应，并通过
    JSON-RPC协议与服务器通信。

    Attributes:
        name (str): MCP服务器名称
        command (str): 启动命令
        args (List[str]): 命令参数列表
        env (Optional[Dict[str, str]]): 环境变量
        process (Optional[subprocess.Popen]): 服务器进程对象
        tools (List[Dict[str, Any]]): 服务器提供的工具列表
        _lock (Lock): 线程锁，用于保护消息发送过程
        _message_id (int): 消息ID计数器，确保请求与响应匹配
        _stdout_queue (Queue): 标准输出消息队列
        _running (bool): 客户端运行状态标志
        _stdout_thread (Thread): 读取标准输出的后台线程
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        request_timeout: float = 5.0,
    ):
        """
        初始化 MCP 客户端

        Args:
            name (str): MCP 服务器名称，用作唯一标识符
            command (str): 启动命令（如 'npx'、'python' 等）
            args (List[str]): 命令参数列表（如 ['@playwright/mcp@latest']）
            env (Optional[Dict[str, str]], optional): 环境变量字典，None表示使用默认环境
            request_timeout (float, optional): 单次 JSON-RPC 请求的超时秒数，默认 5 秒

        Examples:
            >>> client = MCPClient("playwright", "npx", ["@playwright/mcp@latest"])
            >>> client.name
            'playwright'
        """
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.request_timeout = max(0.1, float(request_timeout))
        self.process: subprocess.Popen | None = None
        self.tools: list[dict[str, Any]] = []
        self._lock = Lock()
        self._message_id = 0
        self._stdout_queue: Queue = Queue()
        self._running = False
        self._notification_handler: Callable[[str, dict[str, Any]], None] | None = None

    def start(self) -> bool:
        """
        启动 MCP 服务器进程

        根据配置启动MCP服务器子进程，并初始化与服务器的连接，获取可用工具列表。

        Returns:
            bool: 是否启动成功

        Examples:
            >>> client = MCPClient("test", "echo", ["hello"])
            >>> success = client.start()
            >>> isinstance(success, bool)
            True
        """
        try:
            # 构建完整命令
            full_command = [self.command, *self.args]

            # 准备环境变量（合并当前环境和自定义环境）
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)

            # Windows 平台特殊处理
            is_windows = sys.platform == "win32"

            # 启动子进程
            if is_windows:
                # Windows 需要 shell=True 来找到 npx 等命令
                self.process = subprocess.Popen(
                    " ".join(full_command),  # Windows 下使用字符串命令
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=process_env,
                    shell=True,  # Windows 必需
                )
            else:
                # Unix/Linux/macOS
                self.process = subprocess.Popen(
                    full_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=process_env,
                )

            # 启动输出读取线程
            self._running = True
            self._stdout_thread = Thread(target=self._read_stdout, daemon=True)
            self._stdout_thread.start()

            # 初始化 MCP 连接并获取工具列表
            if not self._initialize():
                self.stop()
                return False

            print(f"[MCP] 服务器 '{self.name}' 启动成功，提供 {len(self.tools)} 个工具")
            return True

        except Exception as e:
            print(f"[MCP] 启动服务器 '{self.name}' 失败: {e}")
            return False

    def stop(self) -> None:
        """
        停止 MCP 服务器进程

        终止MCP服务器子进程并清理相关资源，确保进程被完全停止。

        Examples:
            >>> client = MCPClient("test", "echo", ["hello"])
            >>> client.stop()  # 停止服务器进程
        """
        self._running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        print(f"[MCP] 服务器 '{self.name}' 已停止")

    def _read_stdout(self) -> None:
        """
        后台线程：读取标准输出

        在独立线程中持续读取MCP服务器的标准输出，并将读取到的行放入队列中，
        供主线程处理响应消息使用。该方法在单独的守护线程中运行。
        """
        if not self.process or not self.process.stdout:
            return

        while self._running and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict) and "method" in message and "id" not in message:
                    self._dispatch_notification(str(message["method"]), message.get("params"))
                else:
                    self._stdout_queue.put(message)
            except Exception as e:
                if self._running:
                    print(f"[MCP] 读取服务器输出错误: {e}")
                break

    def _send_message(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        发送 JSON-RPC 消息到 MCP 服务器

        通过标准输入向MCP服务器发送JSON-RPC格式的请求消息，并等待对应的响应。

        Args:
            method (str): JSON-RPC 方法名，如"initialize"、"tools/list"等
            params (Optional[Dict[str, Any]], optional): 请求参数字典

        Returns:
               Optional[Dict[str, Any]]: 响应数据字典，失败时返回None

        Examples:
            >>> client = MCPClient("test", "echo", ["hello"])
            >>> # response = client._send_message("test_method", {"key": "value"})
            >>> # 注意：这个方法通常由其他方法内部调用
        """
        if not self.process or not self.process.stdin:
            return None

        with self._lock:
            self._message_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._message_id,
                "method": method,
            }
            if params:
                message["params"] = params

            try:
                # 发送消息
                # 将消息转为JSON字符串并通过标准输入发送
                self.process.stdin.write(json.dumps(message) + "\n")
                # 刷新缓冲区确保消息立即发送
                self.process.stdin.flush()

                timeout_count = 0
                max_polls = max(1, int(self.request_timeout / 0.1))
                while timeout_count < max_polls:
                    try:
                        queued = self._stdout_queue.get(timeout=0.1)
                        response = json.loads(queued) if isinstance(queued, str) else queued
                        if not isinstance(response, dict):
                            continue

                        if response.get("id") == self._message_id:
                            if "error" in response:
                                print(f"[MCP] 服务器错误: {response['error']}")
                                return None
                            return response.get("result")

                        # Requests are serialized by ``_lock``. An unmatched response belongs
                        # to a stale request, while notifications are dispatched by the reader.
                        continue
                    except Empty:
                        timeout_count += 1
                    except json.JSONDecodeError:
                        continue

                print("[MCP] 响应超时")
                return None

            except Exception as e:
                print(f"[MCP] 发送请求失败: {e}")
                return None

    def _initialize(self) -> bool:
        """
        初始化 MCP 连接并获取工具列表

        发送初始化请求到MCP服务器，建立连接并获取服务器提供的工具列表。

        Returns:
            bool: 是否初始化成功

        Examples:
            >>> client = MCPClient("test", "echo", ["hello"])
            >>> # success = client._initialize()
            >>> # 注意：这个方法通常由start方法内部调用
        """
        # 发送初始化请求
        result = self._send_message(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "dm-code-agent", "version": "1.1.0"},
            },
        )

        if not result:
            return False

        self._send_notification("notifications/initialized")

        # 获取工具列表
        return self.refresh_tools()

    def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification without competing for a response id."""
        if not self.process or not self.process.stdin:
            return
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        try:
            with self._lock:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
        except OSError:
            return

    def refresh_tools(self) -> bool:
        """Reload this server's tool catalog after ``tools/list_changed``."""
        tools_result = self._send_message("tools/list")
        tools = tools_result.get("tools") if isinstance(tools_result, dict) else None
        if not isinstance(tools, list):
            return False
        self.tools = [tool for tool in tools if isinstance(tool, dict)]
        return True

    def set_notification_handler(
        self, handler: Callable[[str, dict[str, Any]], None] | None
    ) -> None:
        """Install a process-wide notification sink for this MCP connection."""
        self._notification_handler = handler

    def _dispatch_notification(self, method: str, params: Any) -> None:
        handler = self._notification_handler
        if handler is None:
            return
        payload = params if isinstance(params, dict) else {}
        # The reader must keep consuming stdout while a refresh waits for tools/list.
        Thread(target=handler, args=(method, payload), daemon=True).start()

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """
        调用 MCP 工具

        向MCP服务器发送工具调用请求，并返回工具执行结果。

        Args:
            tool_name (str): 工具名称
            arguments (Dict[str, Any]): 工具参数字典

        Returns:
            Optional[str]: 工具执行结果文本，失败时返回None

        Examples:
            >>> client = MCPClient("test", "echo", ["hello"])
            >>> # result = client.call_tool("test_tool", {"param": "value"})
            >>> # 注意：需要服务器实际运行才能调用工具
        """
        result = self._send_message("tools/call", {"name": tool_name, "arguments": arguments})

        if result and "content" in result:
            # 提取内容（可能是数组）
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                # 获取第一个内容项的文本
                first_item = content[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    return first_item["text"]
                return str(first_item)
            return str(content)

        return None

    def get_tools(self) -> list[dict[str, Any]]:
        """
        获取此 MCP 服务器提供的工具列表

        返回服务器提供的工具定义列表的副本，确保外部修改不会影响内部状态。

        Returns:
            tools (List[Dict[str, Any]]): 工具定义列表的副本

        Examples:
            >>> client = MCPClient("test", "echo", ["hello"])
            >>> tools = client.get_tools()
            >>> isinstance(tools, list)
            True
        """
        return self.tools.copy()

    def is_running(self) -> bool:
        """
        检查 MCP 服务器是否正在运行

        通过检查子进程是否存在且未终止来判断服务器运行状态。

        Returns:
            bool: 是否运行中

        Examples:
            >>> client = MCPClient("test", "echo", ["hello"])
            >>> running = client.is_running()
            >>> isinstance(running, bool)
            True
        """
        return self.process is not None and self.process.poll() is None


class MCPHTTPClient:
    """MCP client for Streamable HTTP (with legacy SSE response tolerance)."""

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        request_timeout: float = 5.0,
        oauth: bool = False,
        oauth_client_id: str = "",
        oauth_client_secret_env: str = "",
    ):
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.oauth = oauth
        self._authorizer = (
            OAuthAuthorizer(url, oauth_client_id, oauth_client_secret_env) if oauth else None
        )
        self._oauth_header = self._authorizer.authorization_header() if self._authorizer else None
        self.request_timeout = max(0.1, float(request_timeout))
        self.tools: list[dict[str, Any]] = []
        self._message_id = 0
        self._running = False
        self._notification_handler: Callable[[str, dict[str, Any]], None] | None = None
        self._request_lock = Lock()
        self._notification_stop = Event()
        self._notification_thread: Thread | None = None
        self._notification_response: Any | None = None
        self._notification_response_lock = Lock()
        self._server_supports_tool_notifications = False

    def start(self) -> bool:
        result = self._send_message(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "dm-code-agent", "version": "1.1.0"},
            },
        )
        if not isinstance(result, dict):
            return False
        capabilities = result.get("capabilities")
        tools_capability = capabilities.get("tools") if isinstance(capabilities, dict) else None
        self._server_supports_tool_notifications = bool(
            isinstance(tools_capability, dict) and tools_capability.get("listChanged")
        )
        self._send_notification("notifications/initialized")
        if not self.refresh_tools():
            return False
        self._running = True
        self._start_notification_stream()
        return True

    def stop(self) -> None:
        self._running = False
        self._notification_stop.set()
        with self._notification_response_lock:
            response = self._notification_response
            self._notification_response = None
        if response is not None:
            with contextlib.suppress(OSError):
                response.close()

    def is_running(self) -> bool:
        return self._running

    def get_tools(self) -> list[dict[str, Any]]:
        return self.tools.copy()

    def refresh_tools(self) -> bool:
        """Refresh this server's catalog after reconnect or an SSE notification."""
        tools_result = self._send_message("tools/list")
        tools = tools_result.get("tools") if isinstance(tools_result, dict) else None
        if not isinstance(tools, list):
            return False
        self.tools = [tool for tool in tools if isinstance(tool, dict)]
        return True

    def set_notification_handler(
        self, handler: Callable[[str, dict[str, Any]], None] | None
    ) -> None:
        self._notification_handler = handler
        self._start_notification_stream()

    def _start_notification_stream(self) -> None:
        if (
            not self._running
            or not self._server_supports_tool_notifications
            or self._notification_handler is None
            or (self._notification_thread is not None and self._notification_thread.is_alive())
        ):
            return
        self._notification_stop.clear()
        self._notification_thread = Thread(target=self._notification_loop, daemon=True)
        self._notification_thread.start()

    def _notification_loop(self) -> None:
        """Listen for server-initiated JSON-RPC notifications over SSE when supported."""
        while self._running and not self._notification_stop.is_set():
            request = urllib.request.Request(
                self.url,
                headers=self._request_headers(accept="text/event-stream"),
                method="GET",
            )
            try:
                response = urllib.request.urlopen(request, timeout=self.request_timeout)
            except urllib.error.HTTPError:
                # A server that rejects GET does not expose a notification stream.
                return
            except (OSError, urllib.error.URLError):
                self._notification_stop.wait(0.5)
                continue
            try:
                if response.headers.get_content_type() != "text/event-stream":
                    return
                with self._notification_response_lock:
                    self._notification_response = response
                self._consume_notification_stream(response)
            except (OSError, UnicodeError):
                pass
            finally:
                with self._notification_response_lock:
                    if self._notification_response is response:
                        self._notification_response = None
                response.close()
            self._notification_stop.wait(0.5)

    def _consume_notification_stream(self, response: Any) -> None:
        data_lines: list[str] = []
        while self._running and not self._notification_stop.is_set():
            raw = response.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                self._dispatch_sse_event(data_lines)
                data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

    def _dispatch_sse_event(self, data_lines: list[str]) -> None:
        if not data_lines:
            return
        try:
            message = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return
        if not isinstance(message, dict) or "id" in message or "method" not in message:
            return
        handler = self._notification_handler
        if handler is None:
            return
        params = message.get("params")
        payload = params if isinstance(params, dict) else {}
        Thread(target=handler, args=(str(message["method"]), payload), daemon=True).start()

    def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = params
        headers = self._request_headers(content_type="application/json")
        request = urllib.request.Request(
            self.url, data=json.dumps(message).encode(), headers=headers, method="POST"
        )
        try:
            with self._request_lock, urllib.request.urlopen(request, timeout=self.request_timeout):
                return
        except (OSError, urllib.error.URLError):
            return

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        result = self._send_message("tools/call", {"name": tool_name, "arguments": arguments})
        if not isinstance(result, dict):
            return None
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            ]
            return "\n".join(texts)
        return str(content)

    def _send_message(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        with self._request_lock:
            self._message_id += 1
            request_id = self._message_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params

        for attempt in range(2):
            headers = self._request_headers(
                content_type="application/json", accept="application/json, text/event-stream"
            )
            request = urllib.request.Request(
                self.url, data=json.dumps(message).encode(), headers=headers, method="POST"
            )
            try:
                with (
                    self._request_lock,
                    urllib.request.urlopen(request, timeout=self.request_timeout) as response,
                ):
                    body = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as error:
                if error.code == 401 and attempt == 0 and self._begin_oauth(error):
                    continue
                return None
            except (OSError, urllib.error.URLError):
                return None
            if response.headers.get_content_type() == "text/event-stream":
                for line in body.splitlines():
                    if line.startswith("data:"):
                        try:
                            value = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, dict) and value.get("id") == request_id:
                            return value.get("result")
                return None
            try:
                value = json.loads(body)
            except json.JSONDecodeError:
                return None
            return value.get("result") if isinstance(value, dict) and "error" not in value else None
        return None

    def _request_headers(
        self, *, content_type: str | None = None, accept: str | None = None
    ) -> dict[str, str]:
        """Build one authenticated header snapshot for RPC, notifications and SSE."""
        headers = dict(self.headers)
        if content_type:
            headers["Content-Type"] = content_type
        if accept:
            headers["Accept"] = accept
        if self._oauth_header and "Authorization" not in headers:
            headers["Authorization"] = self._oauth_header
        return headers

    def _begin_oauth(self, error: urllib.error.HTTPError) -> bool:
        """Start interactive OAuth only after a protected resource challenges us."""
        if self._authorizer is None:
            return False
        challenge = error.headers.get("WWW-Authenticate", "")
        try:
            self._oauth_header = self._authorizer.authorize(challenge)
        except OAuthError as exc:
            print(f"[MCP] 远程服务器 '{self.name}' OAuth 授权失败: {exc}")
            return False
        return True
