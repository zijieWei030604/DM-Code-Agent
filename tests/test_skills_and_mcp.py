import json
from io import BytesIO
from queue import Queue
from threading import Event
from urllib.error import HTTPError

from dm_agent.core import ReactAgent
from dm_agent.mcp.config import MCPConfig, MCPServerConfig
from dm_agent.skills import ConfigSkill, SkillManager
from dm_agent.skills.selector import SkillSelector
from dm_agent.tools.base import Tool


def test_skill_selector_uses_keywords_and_patterns():
    python_skill = ConfigSkill(
        {
            "name": "python_expert",
            "display_name": "Python Expert",
            "description": "Python help",
            "keywords": ["python", "pytest"],
            "patterns": [r"\.py\b"],
            "priority": 1,
        }
    )
    db_skill = ConfigSkill(
        {
            "name": "db_expert",
            "display_name": "DB Expert",
            "description": "Database help",
            "keywords": ["sql"],
            "priority": 5,
        }
    )

    selector = SkillSelector(max_active_skills=1, min_keyword_score=0.01)
    selected = selector.select(
        "write pytest coverage for app.py",
        {"python_expert": python_skill, "db_expert": db_skill},
    )

    assert selected == ["python_expert"]


def test_skill_manager_loads_custom_json(tmp_path):
    skill_file = tmp_path / "devops.json"
    skill_file.write_text(
        json.dumps(
            {
                "name": "devops_expert",
                "display_name": "DevOps Expert",
                "description": "Docker and CI guidance",
                "keywords": ["docker", "ci"],
                "prompt_addition": "Prefer reproducible deployment steps.",
            }
        ),
        encoding="utf-8",
    )

    manager = SkillManager()
    assert manager.load_custom_skills(tmp_path) == 1
    manager.activate_skills(["devops_expert"])

    assert "devops_expert" in manager.skills
    assert "reproducible deployment" in manager.get_active_prompt_additions()


def test_mcp_config_round_trip_and_enabled_filter():
    config = MCPConfig()
    config.add_server(MCPServerConfig("enabled", "npx", ["tool"], enabled=True))
    config.add_server(MCPServerConfig("disabled", "npx", ["tool"], enabled=False))

    data = config.to_dict()
    restored = MCPConfig.from_dict(data)

    assert set(restored.servers) == {"enabled", "disabled"}
    assert list(restored.get_enabled_servers()) == ["enabled"]


def test_mcp_config_parses_and_round_trips_timeout():
    config = MCPConfig.from_dict(
        {
            "mcpServers": {
                "slow": {"command": "npx", "args": ["tool"], "timeout": 30},
                "default": {"command": "npx", "args": ["tool"]},
            }
        }
    )

    assert config.servers["slow"].timeout == 30.0
    assert config.servers["default"].timeout == 5.0

    data = config.to_dict()
    assert data["mcpServers"]["slow"]["timeout"] == 30.0
    assert "timeout" not in data["mcpServers"]["default"]


def test_mcp_config_round_trips_remote_oauth_flag():
    config = MCPConfig.from_dict(
        {
            "mcpServers": {
                "github": {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp/readonly",
                    "oauth": True,
                    "oauth_client_id": "github-client-id",
                    "oauth_client_secret_env": "GITHUB_MCP_OAUTH_CLIENT_SECRET",
                    "enabled": False,
                }
            }
        }
    )

    assert config.servers["github"].oauth is True
    assert config.to_dict()["mcpServers"]["github"]["oauth"] is True
    assert config.servers["github"].oauth_client_id == "github-client-id"


def test_mcp_tool_wrapper_reconnects_once_when_server_dies():
    from dm_agent.mcp.manager import MCPManager

    class StubClient:
        def __init__(self, *, running=True, result="ok"):
            self.running = running
            self.result = result
            self.calls = 0

        def is_running(self):
            return self.running

        def call_tool(self, tool_name, arguments):
            self.calls += 1
            return self.result

        def stop(self):
            self.running = False

    manager = MCPManager(
        MCPConfig(servers={"srv": MCPServerConfig("srv", "echo", ["hi"], enabled=True)})
    )
    dead = StubClient(running=False)
    fresh = StubClient(running=True, result="reconnected result")
    manager.clients["srv"] = dead

    def fake_start_server(name):
        manager.clients[name] = fresh
        return True

    manager.start_server = fake_start_server  # type: ignore[method-assign]

    tool = manager._create_tool_wrapper(
        server_name="srv", tool_name="ping", description="ping", input_schema={}
    )
    observation = tool.execute({})

    assert observation == "reconnected result"
    assert manager.reconnect_counts["srv"] == 1
    assert fresh.calls == 1
    assert dead.calls == 0


def test_mcp_tool_wrapper_reports_when_reconnect_fails():
    from dm_agent.mcp.manager import MCPManager

    manager = MCPManager(
        MCPConfig(servers={"srv": MCPServerConfig("srv", "echo", ["hi"], enabled=True)})
    )

    manager.start_server = lambda name: False  # type: ignore[method-assign]

    tool = manager._create_tool_wrapper(
        server_name="srv", tool_name="ping", description="ping", input_schema={}
    )
    observation = tool.execute({})

    assert "未运行" in observation
    assert manager.reconnect_counts["srv"] == 1


def test_mcp_client_ignores_notifications_while_waiting_for_response():
    from dm_agent.mcp.client import MCPClient

    class StubStdin:
        def write(self, value):
            return len(value)

        def flush(self):
            pass

    client = MCPClient("srv", "echo", ["hi"])
    client.process = type("Process", (), {"stdin": StubStdin()})()
    client._stdout_queue = Queue()
    client._stdout_queue.put(json.dumps({"jsonrpc": "2.0", "method": "notifications/log"}))
    client._stdout_queue.put(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}))

    assert client._send_message("tools/list") == {"ok": True}


def test_mcp_client_dispatches_server_notifications_without_blocking_reader():
    from dm_agent.mcp.client import MCPClient

    client = MCPClient("srv", "echo", ["hi"])
    received = Event()
    messages = []
    client.set_notification_handler(
        lambda method, params: (messages.append((method, params)), received.set())
    )

    client._dispatch_notification("notifications/tools/list_changed", {"revision": 2})

    assert received.wait(1)
    assert messages == [("notifications/tools/list_changed", {"revision": 2})]


def test_http_mcp_sse_notification_dispatches_tools_list_changed():
    from dm_agent.mcp.client import MCPHTTPClient

    client = MCPHTTPClient("remote", "https://example.invalid/mcp")
    client._running = True
    received = Event()
    messages = []
    client.set_notification_handler(
        lambda method, params: (messages.append((method, params)), received.set())
    )
    stream = BytesIO(
        b"event: notification\n"
        b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed","params":{}}\n\n'
    )

    client._consume_notification_stream(stream)

    assert received.wait(1)
    assert messages == [("notifications/tools/list_changed", {})]


def test_http_mcp_oauth_challenge_obtains_and_uses_bearer_header():
    from dm_agent.mcp.client import MCPHTTPClient

    class StubAuthorizer:
        def __init__(self):
            self.challenges = []

        def authorize(self, challenge):
            self.challenges.append(challenge)
            return "Bearer encrypted-local-token"

    client = MCPHTTPClient("github", "https://example.test/mcp", oauth=False)
    authorizer = StubAuthorizer()
    client._authorizer = authorizer
    error = HTTPError(
        "https://example.test/mcp",
        401,
        "Unauthorized",
        {"WWW-Authenticate": 'Bearer resource_metadata="https://example.test/metadata"'},
        None,
    )

    assert client._begin_oauth(error) is True
    assert client._oauth_header == "Bearer encrypted-local-token"
    assert authorizer.challenges == ['Bearer resource_metadata="https://example.test/metadata"']
    assert client._request_headers(accept="text/event-stream")["Authorization"] == (
        "Bearer encrypted-local-token"
    )


def test_oauth_metadata_and_pkce_helpers_are_deterministic():
    from dm_agent.mcp.oauth import _authorization_metadata_url, _code_challenge, _metadata_url

    assert _metadata_url('Bearer resource_metadata="https://example.test/resource"') == (
        "https://example.test/resource"
    )
    assert _authorization_metadata_url("https://github.com/login/oauth") == (
        "https://github.com/.well-known/oauth-authorization-server/login/oauth"
    )
    assert _code_challenge("test-verifier") == "JBbiqONGWPaAmwXk_8bT6UnlPfrn65D32eZlJS-zGG0"


def test_oauth_uses_pre_registered_client_when_dynamic_registration_is_unavailable(monkeypatch):
    from dm_agent.mcp.oauth import OAuthAuthorizer

    monkeypatch.setenv("GITHUB_MCP_OAUTH_CLIENT_SECRET", "local-secret")
    authorizer = OAuthAuthorizer(
        "https://example.test/mcp",
        client_id="client-id",
        client_secret_env="GITHUB_MCP_OAUTH_CLIENT_SECRET",
    )

    assert authorizer._register(None, "http://127.0.0.1/callback") == {
        "client_id": "client-id",
        "client_secret": "local-secret",
    }


def test_manager_refreshes_changed_server_tools_and_rebinds_live_agent():
    from dm_agent.mcp.manager import MCPManager

    class RefreshingClient:
        def __init__(self):
            self.tools = [{"name": "old", "description": "old", "inputSchema": {}}]

        def is_running(self):
            return True

        def refresh_tools(self):
            self.tools = [{"name": "new", "description": "new", "inputSchema": {}}]
            return True

        def get_tools(self):
            return list(self.tools)

    class FakeRespondClient:
        supports_tool_calling = True

    manager = MCPManager()
    manager.clients["server"] = RefreshingClient()
    manager._rebuild_tools_cache()
    agent = ReactAgent(
        FakeRespondClient(),
        [Tool("read_file", "read", lambda arguments: ""), *manager.get_tools()],
        enable_planning=False,
        enable_compression=False,
    )
    manager.add_tools_changed_listener(agent.refresh_mcp_tools)

    manager._handle_server_notification("server", "notifications/tools/list_changed", {})

    assert "mcp_server_new" in agent.tools
    assert "mcp_server_old" not in agent.tools


def test_prompt_mode_replaces_mcp_catalog_instead_of_accumulating_old_tools():
    class PromptOnlyClient:
        supports_tool_calling = False

    agent = ReactAgent(
        PromptOnlyClient(),
        [Tool("read_file", "read", lambda arguments: "")],
        enable_planning=False,
        enable_compression=False,
    )
    agent.refresh_mcp_tools([Tool("mcp_server_old", "old tool", lambda arguments: "")])
    agent.refresh_mcp_tools([Tool("mcp_server_new", "new tool", lambda arguments: "")])

    assert "mcp_server_new" in agent.system_prompt
    assert "mcp_server_old" not in agent.system_prompt


def test_mcp_client_uses_utf8_for_windows_stdio(monkeypatch):
    from dm_agent.mcp import client as mcp_client

    calls = []

    class StubProcess:
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return 0

        def terminate(self):
            pass

        def wait(self, timeout):
            return 0

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return StubProcess()

    monkeypatch.setattr(mcp_client.sys, "platform", "win32")
    monkeypatch.setattr(mcp_client.subprocess, "Popen", fake_popen)
    client = mcp_client.MCPClient("srv", "npx", ["tool"])
    monkeypatch.setattr(client, "_initialize", lambda: True)

    assert client.start()
    assert calls[0][1]["encoding"] == "utf-8"
    assert calls[0][1]["errors"] == "replace"
