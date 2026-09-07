"""Run a neonize client so that Ctrl-C actually works.

neonize's connect() blocks inside a cgo call into the Go runtime. While the
interpreter sits in that C frame it never executes bytecode, so Python's SIGINT
handler can't run -- Ctrl-C is recorded and then ignored until the call returns,
which it never does. Wrapping connect() in try/except KeyboardInterrupt looks
correct and does nothing.

Fix: run connect() on a daemon thread and idle the main thread in short,
interruptible sleeps. The main thread stays in Python bytecode, so the signal
lands, and daemon=True lets the process exit even though the Go thread is
still parked in the socket loop.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

log = logging.getLogger("runner")

# How long to let a clean disconnect take before killing the process outright.
GRACE_SECONDS = 3.0


def run_until_interrupt(client, on_stop=None) -> None:
    """Block until Ctrl-C, then disconnect cleanly.

    :param client: a neonize NewClient
    :param on_stop: optional callable run before disconnecting (e.g. park the arm)
    """
    thread = threading.Thread(target=client.connect, name="neonize", daemon=True)
    thread.start()

    try:
        while thread.is_alive():
            time.sleep(0.2)         # short enough that Ctrl-C feels instant
        log.info("client stopped on its own")
    except KeyboardInterrupt:
        print()                      # move off the ^C
        log.info("interrupted -- shutting down")
        # Arm the watchdog *before* cleanup: disconnect() is itself a cgo call
        # and can wedge, which would strand us before the hard exit below.
        watchdog = threading.Timer(GRACE_SECONDS, _hard_exit)
        watchdog.daemon = True
        watchdog.start()
        if on_stop is not None:
            try:
                on_stop()
            except Exception:
                log.exception("error during shutdown hook")
        for attempt in (client.disconnect, getattr(client, "stop", None)):
            if attempt is None:
                continue
            try:
                attempt()
                break
            except Exception as exc:
                log.debug("%s failed: %s", getattr(attempt, "__name__", attempt), exc)
        # Don't join the Go thread: if it's wedged in the socket loop it will
        # never return, and daemon=True means we can leave it behind.
        log.info("bye")
        _hard_exit()


def _hard_exit() -> None:
    """Terminate now, without waiting for the Go runtime to unwind.

    daemon=True gets Python to stop waiting on *its* threads, but the Go runtime
    behind cgo owns OS threads that Python doesn't manage and can't kill. A normal
    return from main() therefore prints 'bye' and then hangs forever -- which is
    exactly what it did. os._exit skips interpreter shutdown and ends the process.
    Safe here because everything we care about (the session db) is already flushed
    by the Go side, and stdio is flushed explicitly below.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)
