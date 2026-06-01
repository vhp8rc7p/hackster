#!/usr/bin/env python3
"""
Claude Code PreToolUse hook for Claude Buddy.

Forwards permission requests to the BLE Permission Bridge daemon.
If the daemon is not running, falls through to normal Claude Code behavior.

Tools in DEVICE_TOOLS are sent to the physical device for approval.
All other tools fall through to normal prompting.
"""

import json
import socket
import sys

SOCK_PATH = "/tmp/claude-buddy.sock"

DEVICE_TOOLS = {
    "Bash",
    "Edit",
    "Write",
    "NotebookEdit",
}

WRAP = lambda d: {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": d}}


def main():
    raw = sys.stdin.buffer.read()

    try:
        req = json.loads(raw.decode())
    except Exception:
        json.dump(WRAP("ask"), sys.stdout)
        return

    if req.get("tool_name") not in DEVICE_TOOLS:
        json.dump(WRAP("ask"), sys.stdout)
        return

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(130)
        sock.connect(SOCK_PATH)
        sock.sendall(raw)
        sock.shutdown(socket.SHUT_WR)

        resp = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp += chunk
        sock.close()

        result = json.loads(resp.decode())
        decision = result.get("permissionDecision", "ask")
        json.dump(WRAP(decision), sys.stdout)

    except (ConnectionRefusedError, FileNotFoundError):
        json.dump(WRAP("ask"), sys.stdout)
    except Exception:
        json.dump(WRAP("ask"), sys.stdout)


if __name__ == "__main__":
    main()
