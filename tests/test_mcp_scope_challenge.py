"""Offline coverage for OAuth step-up authorization."""

import json
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from dm_agent.mcp.client import MCPHTTPClient
from dm_agent.mcp.oauth import OAuthAuthorizer


@pytest.mark.parametrize("challenged,expected_requests", [(True, 2), (False, 1)])
def test_http_scope_challenge_retries_once(monkeypatch, challenged, expected_requests):
    client = MCPHTTPClient("test", "https://example.test/mcp")
    requests = []
    challenges = []
    challenge = 'Bearer error="insufficient_scope", scope="repo"' if challenged else ""

    def urlopen(request, timeout):
        requests.append(request)
        raise HTTPError(request.full_url, 403, "Forbidden", {"WWW-Authenticate": challenge}, None)

    def authorize(error):
        challenges.append(error)
        return True

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(client, "_begin_oauth", authorize)
    assert client._send_message("tools/call", {}) is None
    assert len(requests) == expected_requests
    assert len(challenges) == int(challenged)


def test_scope_challenge_success_uses_new_token(monkeypatch):
    client = MCPHTTPClient("test", "https://example.test/mcp")
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise HTTPError(
                request.full_url,
                403,
                "Forbidden",
                {"WWW-Authenticate": 'Bearer error="insufficient_scope", scope="repo"'},
                None,
            )
        response = BytesIO(json.dumps({"result": {"content": [], "isError": False}}).encode())
        response.headers = Message()
        response.headers["Content-Type"] = "application/json"
        return response

    class Authorizer:
        def authorize(self, challenge):
            assert 'scope="repo"' in challenge
            return "Bearer new-test-token"

    client._authorizer = Authorizer()
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert client._send_message("tools/call", {}) == {"content": [], "isError": False}
    assert requests[1].get_header("Authorization") == "Bearer new-test-token"


@pytest.mark.parametrize("scope", ["repo", ""])
def test_authorization_url_preserves_challenged_scope(monkeypatch, scope):
    import dm_agent.mcp.oauth as oauth

    saved = []
    opened = []

    class Store:
        def save(self, token):
            saved.append(token)

    class Callback:
        redirect_uri = "http://127.0.0.1:12345/callback"

        def start(self):
            pass

        def stop(self):
            pass

        def wait_for_code(self, state):
            assert state
            return "test-code"

    metadata = iter(
        [
            {"authorization_servers": ["https://example.test"]},
            {
                "authorization_endpoint": "https://example.test/authorize",
                "token_endpoint": "https://example.test/token",
            },
        ]
    )
    monkeypatch.setattr(oauth, "_get_json", lambda url: next(metadata))
    monkeypatch.setattr(oauth, "_CallbackServer", Callback)
    monkeypatch.setattr(oauth.webbrowser, "open", opened.append)
    monkeypatch.setattr(oauth, "_post_form", lambda *args: {"access_token": "test-token"})
    monkeypatch.setenv("TEST_MCP_SECRET", "test-secret")
    authorizer = OAuthAuthorizer("https://example.test/mcp", "test-id", "TEST_MCP_SECRET", Store())
    challenge = 'Bearer resource_metadata="https://example.test/metadata"'
    if scope:
        challenge += f', scope="{scope}"'
    assert authorizer.authorize(challenge) == "Bearer test-token"
    query = parse_qs(urlsplit(opened[0]).query)
    assert query.get("scope") == ([scope] if scope else None)
    assert saved
