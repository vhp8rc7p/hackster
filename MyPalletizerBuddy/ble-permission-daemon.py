#!/usr/bin/env python3
"""
Claude Buddy BLE Bridge

Bridges Claude Code hooks to the M5Stack Buddy device over BLE (NUS).
Handles permission approval AND activity feed (transcript, tokens).

Start this before using Claude Code:
    python3 ble-permission-daemon.py

Requires: pip install bleak
"""

import asyncio
import json
import os
import sys
import time
import uuid as uuid_mod

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    print("Install bleak: pip install bleak")
    sys.exit(1)

NUS_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX      = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

DEVICE_NAMES  = ("Claude-", "Claude MyPal")
SOCK_PATH     = "/tmp/claude-buddy.sock"
HEARTBEAT_S   = 10
PROMPT_TIMEOUT = 120

pending: dict[str, asyncio.Future] = {}
prompt_lock = asyncio.Lock()
active_prompt_id: str | None = None
rx_buffer = bytearray()

entries: list[str] = []
MAX_ENTRIES = 8
msg = "claude-code"
tokens_total = 0
tokens_today = 0
tokens_day = time.localtime().tm_yday
sessions_running = 0


def add_entry(tool: str, hint: str):
    global msg
    ts = time.strftime("%H:%M")
    short = f"{ts} {tool}"
    if hint:
        room = 88 - len(short) - 1
        if room > 0:
            short += " " + hint[:room]
    entries.insert(0, short)
    while len(entries) > MAX_ENTRIES:
        entries.pop()
    msg = short[:23]


def set_tokens(cumulative: int):
    global tokens_total, tokens_today, tokens_day
    today = time.localtime().tm_yday
    if today != tokens_day:
        tokens_today = 0
        tokens_day = today
    tokens_total = cumulative
    tokens_today = cumulative


def on_ble_notify(_sender, data: bytearray):
    global rx_buffer
    rx_buffer.extend(data)
    while b'\n' in rx_buffer:
        line, rx_buffer = rx_buffer.split(b'\n', 1)
        if not line:
            continue
        try:
            m = json.loads(line.decode('utf-8', errors='replace'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if m.get("cmd") == "permission":
            pid = m.get("id", "")
            decision = m.get("decision", "")
            print(f"[<] response id={pid[:8]}.. decision={decision}")
            if pid in pending and not pending[pid].done():
                pending[pid].set_result(decision)


async def ble_send(client: BleakClient, payload: dict):
    data = json.dumps(payload, separators=(',', ':')) + '\n'
    try:
        await client.write_gatt_char(NUS_RX, data.encode())
    except Exception as e:
        print(f"[!] BLE write failed: {e}")


def extract_hint(tool_input):
    if isinstance(tool_input, dict):
        for key in ("command", "file_path", "query", "url", "prompt", "description"):
            if key in tool_input:
                return str(tool_input[key])
        return json.dumps(tool_input, separators=(',', ':'))
    return str(tool_input)


async def handle_hook(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                      client: BleakClient):
    global active_prompt_id, sessions_running
    try:
        raw = await asyncio.wait_for(reader.read(65536), timeout=5)
        if not raw:
            return

        req = json.loads(raw.decode())
        msg_type = req.get("type", "pre")

        if msg_type == "post":
            tool = req.get("tool", "?")
            hint = req.get("hint", "")
            tok = req.get("tokens", 0)
            add_entry(tool, hint)
            if tok > 0:
                set_tokens(tok)
            print(f"[~] {tool}: {hint[:40]}")
            writer.write(b'{"ok":true}')
            await writer.drain()
            return

        tool_name = req.get("tool_name", "?")[:19]
        tool_input = req.get("tool_input", {})
        hint = extract_hint(tool_input)[:43]
        pid = str(uuid_mod.uuid4())[:36]

        async with prompt_lock:
            active_prompt_id = pid
            sessions_running = 1
            print(f"[>] {tool_name}: {hint[:40]}")

            await ble_send(client, {
                "waiting": 1, "running": 1, "total": 1,
                "msg": f"approve: {tool_name}"[:23],
                "prompt": {"id": pid, "tool": tool_name, "hint": hint}
            })

            fut = asyncio.get_event_loop().create_future()
            pending[pid] = fut
            try:
                decision = await asyncio.wait_for(fut, timeout=PROMPT_TIMEOUT)
            except asyncio.TimeoutError:
                print(f"[!] timeout -> deny")
                decision = "timeout"
            finally:
                pending.pop(pid, None)
                active_prompt_id = None

            await ble_send(client, {"waiting": 0})

        hook_decision = "allow" if decision == "once" else "deny"
        print(f"[=] {decision} -> {hook_decision}")
        writer.write(json.dumps({"permissionDecision": hook_decision}).encode())
        await writer.drain()

    except Exception as e:
        print(f"[!] hook error: {e}")
        writer.write(json.dumps({"permissionDecision": "ask"}).encode())
        await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def heartbeat(client: BleakClient):
    while client.is_connected:
        if active_prompt_id is None:
            payload = {
                "total": 1,
                "running": sessions_running,
                "waiting": 0,
                "msg": msg,
                "tokens_today": tokens_today,
            }
            if entries:
                payload["entries"] = entries[:MAX_ENTRIES]
            await ble_send(client, payload)
        await asyncio.sleep(HEARTBEAT_S)


async def find_device():
    print(f"[*] Scanning for {DEVICE_NAMES}...")
    devices = await BleakScanner.discover(timeout=10, return_adv=True)
    for addr, (d, adv) in devices.items():
        name = d.name or getattr(adv, 'local_name', '') or ""
        if any(name.startswith(p) or name == p for p in DEVICE_NAMES):
            print(f"[+] Found: {name} ({d.address})")
            return d
    print("[-] Not found")
    return None


async def main():
    global rx_buffer

    if os.path.exists(SOCK_PATH):
        os.unlink(SOCK_PATH)

    while True:
        device = await find_device()
        if not device:
            await asyncio.sleep(5)
            continue

        rx_buffer = bytearray()

        try:
            async with BleakClient(device, timeout=15) as client:
                print(f"[+] Connected: {device.name}")
                await client.start_notify(NUS_TX, on_ble_notify)

                server = await asyncio.start_unix_server(
                    lambda r, w: handle_hook(r, w, client),
                    path=SOCK_PATH
                )
                os.chmod(SOCK_PATH, 0o600)
                print(f"[+] Socket: {SOCK_PATH}")
                print(f"[+] Ready")

                hb = asyncio.create_task(heartbeat(client))
                try:
                    while client.is_connected:
                        await asyncio.sleep(1)
                finally:
                    hb.cancel()
                    server.close()
                    await server.wait_closed()
                    for pid, fut in list(pending.items()):
                        if not fut.done():
                            fut.set_result("disconnect")
                    pending.clear()

        except Exception as e:
            print(f"[!] {e}")

        if os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)
        print("[*] Reconnecting in 3s...")
        await asyncio.sleep(3)


if __name__ == "__main__":
    print("=== Claude Buddy BLE Bridge ===")
    print(f"Socket: {SOCK_PATH}\n")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Bye")
        if os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)
