"""Small synchronous JSON-RPC client for a local LSP server.

The client intentionally owns no repository index.  Pyright remains the source of
symbol, reference, and diagnostic facts; when it is absent callers receive an
explicit unavailable result instead of falling back to the legacy AST workspace.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlparse


def path_to_uri(path: Path) -> str:
    # Keep the Windows drive separator literal: ``file:///E:/repo/a.py`` is the
    # URI form expected by Pyright, whereas encoding ``:`` produces a different path.
    return "file:///" + quote(path.resolve().as_posix(), safe="/:")


def _document_key(uri: str) -> str:
    path = unquote(urlparse(uri).path)
    return path.casefold() if os.name == "nt" else path


class LspClient(Protocol):
    """The narrow LSP surface used by impact analysis and deterministic fakes."""

    available: bool
    unavailable_reason: str

    def start(self, workspace_root: Path) -> bool: ...

    def close(self) -> None: ...

    def document_symbols(self, path: Path, text: str) -> list[dict[str, Any]]: ...

    def references(
        self, path: Path, line: int, character: int, text: str
    ) -> list[dict[str, Any]]: ...

    def definition(
        self, path: Path, line: int, character: int, text: str
    ) -> list[dict[str, Any]]: ...

    def diagnostics(self, path: Path, text: str) -> list[dict[str, Any]]: ...


class PyrightLspClient:
    """A bounded stdio client for ``pyright-langserver --stdio``."""

    def __init__(self, command: str = "pyright-langserver", timeout_seconds: float = 5.0) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.available = False
        self.unavailable_reason = "not started"
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._notifications: dict[str, list[dict[str, Any]]] = {}
        self._pending_diagnostics: dict[str, int] = {}
        self._document_versions: dict[Path, int] = {}
        self._notification_condition = threading.Condition()
        self._reader: threading.Thread | None = None
        self._next_id = 0
        self._opened: dict[Path, str] = {}

    def start(self, workspace_root: Path) -> bool:
        executable = shutil.which(self.command)
        if executable is None:
            self.unavailable_reason = f"LSP command not found: {self.command}"
            return False
        try:
            self._process = subprocess.Popen(
                [executable, "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
            self._request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": path_to_uri(workspace_root),
                    "capabilities": {
                        "textDocument": {
                            "publishDiagnostics": {"relatedInformation": True},
                            "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        }
                    },
                    "workspaceFolders": [
                        {"uri": path_to_uri(workspace_root), "name": workspace_root.name}
                    ],
                },
            )
            self._notify("initialized", {})
            self.available = True
            self.unavailable_reason = ""
            return True
        except (OSError, RuntimeError, TimeoutError) as error:
            self.unavailable_reason = f"LSP start failed: {error}"
            self.close()
            return False

    def close(self) -> None:
        process = self._process
        self._process = None
        self.available = False
        if process is None:
            return
        try:
            if process.poll() is None:
                self._notify("shutdown", {})
                process.terminate()
                process.wait(timeout=1)
        except (OSError, subprocess.SubprocessError):
            pass

    def document_symbols(self, path: Path, text: str) -> list[dict[str, Any]]:
        self._sync(path, text)
        result = self._request(
            "textDocument/documentSymbol", {"textDocument": {"uri": path_to_uri(path)}}
        )
        return list(result) if isinstance(result, list) else []

    def references(self, path: Path, line: int, character: int, text: str) -> list[dict[str, Any]]:
        self._sync(path, text)
        result = self._request(
            "textDocument/references",
            {
                "textDocument": {"uri": path_to_uri(path)},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": True},
            },
        )
        return list(result) if isinstance(result, list) else []

    def definition(self, path: Path, line: int, character: int, text: str) -> list[dict[str, Any]]:
        self._sync(path, text)
        result = self._request(
            "textDocument/definition",
            {
                "textDocument": {"uri": path_to_uri(path)},
                "position": {"line": line, "character": character},
            },
        )
        return (
            list(result)
            if isinstance(result, list)
            else ([result] if isinstance(result, dict) else [])
        )

    def diagnostics(self, path: Path, text: str) -> list[dict[str, Any]]:
        self._sync(path, text)
        # Pyright publishes diagnostics asynchronously.  Wait for the document version we
        # just sent rather than reading a possibly stale cache after a fixed sleep.
        uri = _document_key(path_to_uri(path))
        deadline = time.monotonic() + self.timeout_seconds
        with self._notification_condition:
            while uri in self._pending_diagnostics and time.monotonic() < deadline:
                self._notification_condition.wait(timeout=max(0.01, deadline - time.monotonic()))
            if uri in self._pending_diagnostics:
                raise TimeoutError(f"LSP diagnostics timed out for {path}")
            return list(self._notifications.get(uri, []))

    def _sync(self, path: Path, text: str) -> None:
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "LSP is unavailable")
        uri = path_to_uri(path)
        previous = self._opened.get(path)
        if previous == text:
            return
        version = self._document_versions.get(path, 0) + 1
        with self._notification_condition:
            self._pending_diagnostics[_document_key(uri)] = version
        if previous is None:
            version = 1
            self._notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "python",
                        "version": version,
                        "text": text,
                    }
                },
            )
        elif previous != text:
            version = self._document_versions.get(path, 1) + 1
            self._notify(
                "textDocument/didChange",
                {
                    "textDocument": {
                        "uri": uri,
                        "version": version,
                    },
                    "contentChanges": [{"text": text}],
                },
            )
        else:
            return
        self._document_versions[path] = version
        self._opened[path] = text

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        if self._process is None:
            raise RuntimeError("LSP is not running")
        self._next_id += 1
        request_id = self._next_id
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + self.timeout_seconds
        deferred: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            try:
                message = self._responses.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if message.get("id") == request_id:
                for item in deferred:
                    self._responses.put(item)
                if "error" in message:
                    raise RuntimeError(str(message["error"]))
                return message.get("result")
            deferred.append(message)
        for item in deferred:
            self._responses.put(item)
        raise TimeoutError(f"LSP request timed out: {method}")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        if self._process is not None:
            self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _write(self, message: Mapping[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("LSP stdin is unavailable")
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        self._process.stdin.write(
            f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload
        )
        self._process.stdin.flush()

    def _read_loop(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        stream = self._process.stdout
        while True:
            headers: dict[bytes, bytes] = {}
            while True:
                line = stream.readline()
                if not line:
                    return
                line = line.strip()
                if not line:
                    break
                key, _, value = line.partition(b":")
                headers[key.lower()] = value.strip()
            length = int(headers.get(b"content-length", b"0"))
            if length <= 0:
                continue
            try:
                message = json.loads(stream.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if message.get("method") == "textDocument/publishDiagnostics":
                params = message.get("params") or {}
                uri = _document_key(str(params.get("uri", "")))
                values = params.get("diagnostics")
                with self._notification_condition:
                    self._notifications[uri] = list(values) if isinstance(values, Sequence) else []
                    expected = self._pending_diagnostics.get(uri)
                    version = params.get("version")
                    if expected is not None and (
                        not isinstance(version, int) or version >= expected
                    ):
                        self._pending_diagnostics.pop(uri, None)
                    self._notification_condition.notify_all()
            elif "id" in message:
                self._responses.put(message)
