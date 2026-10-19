"""Run one read-only live MCP call and show its unprocessed response."""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dm_agent.mcp.client import MCPHTTPClient
from dm_agent.mcp.config import load_mcp_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--error-demo",
        action="store_true",
        help="Omit the required query to inspect the server's raw error response.",
    )
    args = parser.parse_args()
    config = load_mcp_config().servers["github_readonly"]
    client = MCPHTTPClient(
        config.name,
        config.url,
        headers=config.headers,
        request_timeout=30,
        oauth=config.oauth,
        oauth_client_id=config.oauth_client_id,
        oauth_client_secret_env=config.oauth_client_secret_env,
    )
    try:
        if not client.start():
            print("MCP initialization failed")
            return
        tool = next(t for t in client.tools if t["name"] == "search_repositories")
        print("TOOL SCHEMA:", json.dumps(tool.get("inputSchema"), ensure_ascii=False))
        arguments = {} if args.error_demo else {"query": "repo:can1357/oh-my-pi", "perPage": 1}
        message = {
            "jsonrpc": "2.0",
            "id": 10001,
            "method": "tools/call",
            "params": {
                "name": "search_repositories",
                "arguments": arguments,
            },
        }
        print("CALL PARAMS:", json.dumps(message["params"], ensure_ascii=False))
        request = urllib.request.Request(
            config.url,
            data=json.dumps(message).encode(),
            headers=client._request_headers(
                content_type="application/json", accept="application/json, text/event-stream"
            ),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                print("HTTP STATUS:", response.status)
                print("CONTENT TYPE:", response.headers.get("Content-Type"))
                print("RAW BODY:", response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as error:
            print("HTTP STATUS:", error.code)
            print("WWW-AUTHENTICATE:", error.headers.get("WWW-Authenticate", "(absent)"))
            print(
                "ACCEPTED OAUTH SCOPES:", error.headers.get("X-Accepted-OAuth-Scopes", "(absent)")
            )
            print("RAW ERROR BODY:", error.read().decode("utf-8", errors="replace"))
            if error.code == 403 and 'error="insufficient_scope"' in error.headers.get(
                "WWW-Authenticate", ""
            ):
                print("Requesting additional authorization in your browser...")
                if client._begin_oauth(error):
                    result = client._send_message("tools/call", message["params"])
                    print("RAW MCP RESULT:", json.dumps(result, ensure_ascii=False, indent=2))
        except (OSError, urllib.error.URLError) as error:
            print("TRANSPORT ERROR:", type(error).__name__, str(error))
    finally:
        client.stop()


if __name__ == "__main__":
    main()
