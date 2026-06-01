#!/usr/bin/env python3
"""
Claude Code PostToolUse hook for Claude Buddy.

Sends tool completion events to the BLE bridge daemon for the
transcript feed and real token counting from the session transcript.
"""

import json
import os
import socket
import sys

SOCK_PATH = "/tmp/claude-buddy.sock"
TOKEN_CACHE = "/tmp/claude-buddy-tokens.json"


def get_session_tokens(transcript_path, session_id):
    """Incrementally read output tokens from transcript, caching progress."""
    if not transcript_path or not os.path.exists(transcript_path):
        return 0

    offset = 0
    total = 0
    cached_session = ""

    try:
        with open(TOKEN_CACHE) as f:
            cache = json.load(f)
            cached_session = cache.get("sid", "")
            if cached_session == session_id:
                offset = cache.get("off", 0)
                total = cache.get("tok", 0)
    except Exception:
        pass

    if cached_session != session_id:
        offset = 0
        total = 0

    try:
        size = os.path.getsize(transcript_path)
        if offset >= size:
            return total

        with open(transcript_path) as f:
            f.seek(offset)
            for line in f:
                if '"output_tokens"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("type") == "assistant":
                    usage = obj.get("message", {}).get("usage", {})
                    total += usage.get("output_tokens", 0)
            new_offset = f.tell()

        with open(TOKEN_CACHE, 'w') as f:
            json.dump({"sid": session_id, "off": new_offset, "tok": total}, f)

    except Exception:
        pass

    return total


def main():
    raw = sys.stdin.buffer.read()

    try:
        req = json.loads(raw.decode())
    except Exception:
        return

    tool = req.get("tool_name", "")
    tool_input = req.get("tool_input", {})
    transcript = req.get("transcript_path", "")
    session_id = req.get("session_id", "")

    if isinstance(tool_input, dict):
        for key in ("command", "file_path", "query", "url", "prompt", "description"):
            if key in tool_input:
                hint = str(tool_input[key])
                break
        else:
            hint = ""
    else:
        hint = str(tool_input)

    hint = hint.split('\n')[0][:80]
    tokens = get_session_tokens(transcript, session_id)

    msg = json.dumps({
        "type": "post",
        "tool": tool[:19],
        "hint": hint,
        "tokens": tokens,
    })

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(SOCK_PATH)
        sock.sendall(msg.encode())
        sock.shutdown(socket.SHUT_WR)
        sock.recv(256)
        sock.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
