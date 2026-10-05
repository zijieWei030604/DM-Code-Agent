"""OAuth 2.1 helpers for protected remote MCP servers.

The module intentionally uses only the standard library.  Remote MCP credentials
are encrypted with Windows DPAPI before they leave process memory; the repository
never receives a token or a client registration record.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import secrets
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event, Thread
from typing import Any


class OAuthError(RuntimeError):
    """A remote MCP authorization flow could not be completed."""


@dataclass(frozen=True)
class OAuthToken:
    """Persisted OAuth registration and token material for one MCP resource."""

    access_token: str
    refresh_token: str | None
    expires_at: float | None
    client_id: str
    client_secret: str | None
    token_endpoint: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OAuthToken | None:
        access_token = value.get("access_token")
        client_id = value.get("client_id")
        token_endpoint = value.get("token_endpoint")
        if not all(
            isinstance(item, str) and item for item in (access_token, client_id, token_endpoint)
        ):
            return None
        expires_at = value.get("expires_at")
        return cls(
            access_token=access_token,
            refresh_token=(
                value.get("refresh_token") if isinstance(value.get("refresh_token"), str) else None
            ),
            expires_at=float(expires_at) if isinstance(expires_at, (int, float)) else None,
            client_id=client_id,
            client_secret=(
                value.get("client_secret") if isinstance(value.get("client_secret"), str) else None
            ),
            token_endpoint=token_endpoint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "token_endpoint": self.token_endpoint,
        }


class OAuthTokenStore:
    """A per-user DPAPI-backed store; no plaintext fallback is allowed."""

    def __init__(self, resource_url: str, root: Path | None = None):
        if sys.platform != "win32":
            raise OAuthError("当前 OAuth 凭证存储仅支持 Windows DPAPI")
        digest = hashlib.sha256(resource_url.encode("utf-8")).hexdigest()
        base = (
            root
            or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "dm-code-agent" / "mcp-oauth"
        )
        self.path = base / f"{digest}.bin"

    def load(self) -> OAuthToken | None:
        if not self.path.is_file():
            return None
        try:
            value = json.loads(_unprotect(self.path.read_bytes()).decode("utf-8"))
        except (OSError, UnicodeError, ValueError, OAuthError):
            return None
        return OAuthToken.from_dict(value) if isinstance(value, dict) else None

    def save(self, token: OAuthToken) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _protect(json.dumps(token.to_dict(), separators=(",", ":")).encode("utf-8"))
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, self.path)


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _as_blob(value: bytes) -> tuple[_DataBlob, Any]:
    buffer = (ctypes.c_byte * len(value)).from_buffer_copy(value)
    return _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer


def _protect(value: bytes) -> bytes:
    source, source_buffer = _as_blob(value)
    result = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    if not crypt32.CryptProtectData(
        ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)
    ):
        raise OAuthError("无法使用 Windows DPAPI 加密 OAuth 凭证")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(result.pbData)
        del source_buffer


def _unprotect(value: bytes) -> bytes:
    source, source_buffer = _as_blob(value)
    result = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)
    ):
        raise OAuthError("无法使用 Windows DPAPI 解密 OAuth 凭证")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(result.pbData)
        del source_buffer


class OAuthAuthorizer:
    """Discovers an MCP authorization server and runs authorization-code PKCE."""

    def __init__(
        self,
        resource_url: str,
        client_id: str = "",
        client_secret_env: str = "",
        token_store: OAuthTokenStore | None = None,
    ):
        self.resource_url = resource_url
        self.client_id = client_id
        self.client_secret_env = client_secret_env
        self.token_store = token_store or OAuthTokenStore(resource_url)

    def authorization_header(self) -> str | None:
        token = self.token_store.load()
        if token is None:
            return None
        if token.expires_at is None or token.expires_at > time.time() + 60:
            return f"Bearer {token.access_token}"
        refreshed = self._refresh(token)
        return f"Bearer {refreshed.access_token}" if refreshed is not None else None

    def authorize(self, challenge: str) -> str:
        metadata_url = _metadata_url(challenge)
        resource_metadata = _get_json(metadata_url)
        authorization_servers = resource_metadata.get("authorization_servers")
        if not isinstance(authorization_servers, list) or not authorization_servers:
            raise OAuthError("MCP 服务未提供 OAuth 授权服务器")
        issuer = authorization_servers[0]
        if not isinstance(issuer, str) or not issuer.startswith("https://"):
            raise OAuthError("MCP 服务提供的 OAuth 授权服务器无效")
        server_metadata = _get_json(_authorization_metadata_url(issuer))
        registration_endpoint = server_metadata.get("registration_endpoint")
        authorization_endpoint = server_metadata.get("authorization_endpoint")
        token_endpoint = server_metadata.get("token_endpoint")
        if not all(
            isinstance(item, str) and item for item in (authorization_endpoint, token_endpoint)
        ):
            raise OAuthError("OAuth 授权服务器缺少授权或换取 Token 的端点")

        callback = _CallbackServer()
        callback.start()
        try:
            registration = self._register(registration_endpoint, callback.redirect_uri)
            verifier = _code_verifier()
            state = secrets.token_urlsafe(24)
            query = urllib.parse.urlencode(
                {
                    "response_type": "code",
                    "client_id": registration["client_id"],
                    "redirect_uri": callback.redirect_uri,
                    "code_challenge": _code_challenge(verifier),
                    "code_challenge_method": "S256",
                    "state": state,
                    "resource": resource_metadata.get("resource", self.resource_url),
                }
            )
            webbrowser.open(f"{authorization_endpoint}?{query}")
            code = callback.wait_for_code(state)
            token_payload = _post_form(
                token_endpoint,
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": callback.redirect_uri,
                    "client_id": registration["client_id"],
                    "client_secret": registration.get("client_secret", ""),
                    "code_verifier": verifier,
                    "resource": resource_metadata.get("resource", self.resource_url),
                },
            )
        finally:
            callback.stop()
        token = _token_from_payload(token_payload, registration, token_endpoint)
        self.token_store.save(token)
        return f"Bearer {token.access_token}"

    def _register(self, endpoint: Any, redirect_uri: str) -> dict[str, str]:
        if self.client_id:
            secret = os.environ.get(self.client_secret_env, "") if self.client_secret_env else ""
            if not secret:
                raise OAuthError("预注册 OAuth App 缺少配置的客户端密钥环境变量")
            return {"client_id": self.client_id, "client_secret": secret}
        if not isinstance(endpoint, str) or not endpoint:
            raise OAuthError("授权服务器不支持动态客户端注册；请改用已注册 OAuth 客户端")
        payload = _post_json(
            endpoint,
            {
                "client_name": "DM-Code-Agent",
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        client_id = payload.get("client_id")
        if not isinstance(client_id, str) or not client_id:
            raise OAuthError("动态客户端注册没有返回 client_id")
        secret = payload.get("client_secret")
        return {"client_id": client_id, "client_secret": secret if isinstance(secret, str) else ""}

    def _refresh(self, token: OAuthToken) -> OAuthToken | None:
        if not token.refresh_token:
            return None
        try:
            payload = _post_form(
                token.token_endpoint,
                {
                    "grant_type": "refresh_token",
                    "refresh_token": token.refresh_token,
                    "client_id": token.client_id,
                    "client_secret": token.client_secret or "",
                    "resource": self.resource_url,
                },
            )
            refreshed = _token_from_payload(
                payload,
                {"client_id": token.client_id, "client_secret": token.client_secret or ""},
                token.token_endpoint,
                fallback_refresh_token=token.refresh_token,
            )
            self.token_store.save(refreshed)
            return refreshed
        except (OAuthError, OSError, ValueError):
            return None


class _CallbackServer:
    def __init__(self):
        self.event = Event()
        self.query: dict[str, list[str]] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                outer.event.set()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write("<h1>GitHub 授权完成，可返回 DM-Code-Agent。</h1>".encode())

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/callback"

    def start(self) -> None:
        self.thread.start()

    def wait_for_code(self, expected_state: str) -> str:
        if not self.event.wait(300):
            raise OAuthError("等待 GitHub 授权超时")
        if self.query.get("state", [None])[0] != expected_state:
            raise OAuthError("OAuth 回调 state 不匹配")
        if self.query.get("error"):
            raise OAuthError(f"GitHub 拒绝授权：{self.query['error'][0]}")
        code = self.query.get("code", [None])[0]
        if not isinstance(code, str) or not code:
            raise OAuthError("OAuth 回调未包含授权码")
        return code

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


def _metadata_url(challenge: str) -> str:
    match = re.search(r'resource_metadata="([^"]+)"', challenge)
    if not match:
        raise OAuthError("远程 MCP 未提供 resource_metadata，无法启动 OAuth")
    return match.group(1)


def _authorization_metadata_url(issuer: str) -> str:
    parsed = urllib.parse.urlsplit(issuer)
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            "/.well-known/oauth-authorization-server" + parsed.path,
            "",
            "",
        )
    )


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise OAuthError("OAuth 元数据不是 JSON 对象")
    return value


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise OAuthError("OAuth 注册响应不是 JSON 对象")
    return value


def _post_form(url: str, payload: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise OAuthError("OAuth Token 响应不是 JSON 对象")
    return value


def _token_from_payload(
    payload: dict[str, Any],
    registration: dict[str, str],
    token_endpoint: str,
    fallback_refresh_token: str | None = None,
) -> OAuthToken:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise OAuthError("OAuth Token 响应未包含 access_token")
    expires_in = payload.get("expires_in")
    expires_at = time.time() + float(expires_in) if isinstance(expires_in, (int, float)) else None
    refresh_token = payload.get("refresh_token")
    return OAuthToken(
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else fallback_refresh_token,
        expires_at=expires_at,
        client_id=registration["client_id"],
        client_secret=registration.get("client_secret") or None,
        token_endpoint=token_endpoint,
    )


def _code_verifier() -> str:
    return secrets.token_urlsafe(64)


def _code_challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
