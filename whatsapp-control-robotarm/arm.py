"""Serial driver for the myCobot 280, fronted by a single-threaded work queue.

Why the queue: neonize dispatches every incoming message on its own thread, and
pyserial talking to the M5Stack is emphatically not safe to hit from several
threads at once. One worker thread owns the serial port for the whole process;
message handlers only ever push Command objects at it. It also means a slow
10-second move can't stall message reception.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import config
from commands import Command

log = logging.getLogger("arm")

Reply = Callable[[str], None]


class ArmController:
    def __init__(self, port: str = config.PORT, baud: int = config.BAUD,
                 dry_run: bool = False):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.speed = config.DEFAULT_SPEED
        self._mc = None
        self._queue: "queue.Queue[tuple[Command, Reply]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        # Set by `stop` so a multi-step routine aborts between steps instead of
        # running to completion after the user has asked it to quit.
        self._cancel = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self.dry_run:
            log.warning("dry run: no serial port will be opened")
        else:
            from pymycobot import MyCobot280
            log.info("opening %s @ %s", self.port, self.baud)
            self._mc = MyCobot280(self.port, self.baud)
            time.sleep(0.5)             # M5Stack needs a moment after port open
            self._mc.power_on()
            self._set_color(config.IDLE_COLOR)
        self._running.set()
        self._thread = threading.Thread(target=self._worker, name="arm", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._running.clear()
        if self._mc is not None:
            try:
                self._mc.set_color(255, 0, 0)
            except Exception:
                pass

    # -- public API --------------------------------------------------------

    def submit(self, cmd: Command, reply: Reply) -> None:
        """Queue a command. Returns immediately; `reply` is called from the worker."""
        if cmd.kind == "stop":
            self._cancel.set()          # aborts a routine already in progress
            dropped = self._drain()
            reply(f"stopping. dropped {dropped} queued move(s)")
            return
        depth = self._queue.qsize()
        self._queue.put((cmd, reply))
        if depth:
            reply(f"queued behind {depth} move(s)")

    def _drain(self) -> int:
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                dropped += 1
            except queue.Empty:
                return dropped

    # -- worker ------------------------------------------------------------

    def _worker(self) -> None:
        while self._running.is_set():
            try:
                cmd, reply = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._cancel.clear()        # a fresh command isn't pre-cancelled
            try:
                reply(self._execute(cmd))
            except Exception as exc:                    # never kill the worker
                log.exception("command failed")
                reply(f"arm error: {exc}")
            finally:
                self._queue.task_done()

    def _execute(self, cmd: Command) -> str:
        kind, args = cmd.kind, cmd.args

        if kind == "speed":
            self.speed = args["value"]
            return f"speed set to {self.speed}"

        if kind == "preset":
            angles = config.PRESETS[args["name"]]
            self._send_angles(angles)
            return f"moved to '{args['name']}'"

        if kind == "routine":
            name = args["name"]
            steps = config.ROUTINES[name]
            try:
                for n, (angles, speed) in enumerate(steps, 1):
                    if self._cancel.is_set():
                        return f"'{name}' stopped after step {n - 1}/{len(steps)}"
                    self._set_color_step(n - 1)
                    self._send_angles(angles, speed=speed)
            finally:
                self._set_color(config.IDLE_COLOR)
            return f"'{name}' done ({len(steps)} steps)"

        if kind == "angles":
            self._send_angles(args["values"])
            return "angles set: " + " ".join(f"{v:g}" for v in args["values"])

        if kind == "coords":
            self._send_coords(args["values"])
            x, y, z = args["values"][:3]
            return f"moved to x={x:g} y={y:g} z={z:g}"

        if kind == "relative":
            coords = self._read_coords()
            if coords is None:
                return "couldn't read current position -- try an absolute 'move' instead"
            axis, delta = args["axis"], args["delta"]
            coords[axis] += delta
            self._send_coords(coords)
            label = "xyz"[axis]
            return f"{label} {delta:+g}mm -> {coords[axis]:.1f}"

        if kind == "pickplace":
            return self._pick_and_place()

        if kind == "pickpoint":
            p = config.PICK_PLACE
            speed = config.PICK_PLACE_SPEED
            self._gripper(config.GRIPPER_OPEN)
            self._send_angles(p["start_angles"], speed)
            coords = self._read_coords()
            if coords is None:
                return "reached the approach pose but couldn't read coordinates"
            if p["pick_z"] >= coords[2]:
                return (f"refusing: pick_z={p['pick_z']} is not below the approach "
                        f"height ({coords[2]:.0f})")
            coords = [float(v) for v in coords]
            coords[2] = float(p["pick_z"])
            self._send_coords(coords, speed)
            final = self._read_coords() or coords
            return (f"at pick point, gripper open -- "
                    f"x={final[0]:.0f} y={final[1]:.0f} z={final[2]:.0f}")

        if kind == "gripper":
            self._gripper(args["value"])
            return f"gripper -> {args['value']}"

        if kind == "relax":
            if not self.dry_run:
                self._mc.release_all_servos()
            return "servos released -- hold the arm, it will drop"

        if kind == "query":
            angles = self._read(lambda: self._mc.get_angles())
            coords = self._read_coords()
            if angles is None and coords is None:
                return "arm isn't answering"
            a = " ".join(f"{v:.0f}" for v in angles) if angles else "?"
            c = " ".join(f"{v:.0f}" for v in coords[:3]) if coords else "?"
            return f"angles: {a}\ncoords (xyz): {c}\nspeed: {self.speed}"

        return f"unhandled command kind '{kind}'"

    # -- serial helpers ----------------------------------------------------

    def _pick_and_place(self) -> str:
        """Angle approach, then single-axis Cartesian moves so the wrist keeps the
        orientation set by start_angles. Only one axis changes per move, which
        makes the path predictable and easy to reason about when it goes wrong."""
        p = config.PICK_PLACE
        speed = config.PICK_PLACE_SPEED
        total = 8
        step = 0

        def checkpoint(label: str) -> Optional[str]:
            nonlocal step
            step += 1
            if self._cancel.is_set():
                return (f"stopped at step {step}/{total} ({label})"
                        + (" -- cube may still be gripped" if step >= 3 else ""))
            log.info("pick-and-place %d/%d: %s", step, total, label)
            return None

        # 1. approach pose, jaws open
        if (msg := checkpoint("approach + open gripper")):
            return msg
        self._gripper(config.GRIPPER_OPEN)
        self._send_angles(p["start_angles"], speed)

        coords = self._read_coords()
        if coords is None:
            return ("reached the approach pose but couldn't read coordinates, "
                    "so I stopped rather than move blind")
        coords = [float(v) for v in coords]
        travel_z = p["travel_z"] if p["travel_z"] is not None else coords[2]
        log.info("pick-and-place: start xyz=(%.1f, %.1f, %.1f), travel_z=%.1f",
                 coords[0], coords[1], coords[2], travel_z)

        # A "descend" that would raise the arm means the configured heights don't
        # match reality -- refuse rather than drive somewhere unexpected.
        for label, z in (("pick_z", p["pick_z"]), ("place_z", p["place_z"])):
            if z >= travel_z:
                return (f"refusing: {label}={z} is not below the travel height "
                        f"({travel_z:.0f}). Descending would move the arm UP. "
                        f"Lower {label} in config.py, or set travel_z explicitly.")

        def move_axis(axis: int, value: float) -> None:
            coords[axis] = float(value)
            self._send_coords(list(coords), speed)

        # 2-4. descend, grip, lift
        if (msg := checkpoint(f"descend to z={p['pick_z']}")):
            return msg
        move_axis(2, p["pick_z"])

        if (msg := checkpoint("close gripper")):
            return msg
        self._gripper(config.GRIPPER_CLOSED)

        if (msg := checkpoint(f"ascend to z={travel_z:.0f}")):
            return msg
        move_axis(2, travel_z)

        # 5-7. travel, descend, release
        if (msg := checkpoint(f"slide to y={p['place_y']}")):
            return msg
        move_axis(1, p["place_y"])

        if (msg := checkpoint(f"descend to z={p['place_z']}")):
            return msg
        move_axis(2, p["place_z"])

        if (msg := checkpoint("open gripper -- drop in box")):
            return msg
        self._gripper(config.GRIPPER_OPEN)

        # 8. retreat
        if (msg := checkpoint(f"ascend to z={travel_z:.0f}")):
            return msg
        move_axis(2, travel_z)

        return "pick-and-place done -- cube should be in the box"

    def _set_color(self, rgb) -> None:
        """Never let a decorative LED call break a motion sequence."""
        if self.dry_run or self._mc is None:
            log.debug("led -> %s", tuple(rgb))
            return
        try:
            self._mc.set_color(*(int(v) for v in rgb))
        except Exception as exc:
            log.debug("set_color failed: %s", exc)

    def _set_color_step(self, index: int) -> None:
        # Drop (0,0,0) entries: an "off" LED mid-routine reads as the arm having
        # died rather than as a colour, so it's never what you want here.
        palette = [c for c in config.ROUTINE_COLORS if tuple(c) != (0, 0, 0)]
        if not palette:
            return
        self._set_color(palette[index % len(palette)])

    def _gripper(self, value: int) -> None:
        log.info("gripper -> %s", value)
        if self.dry_run:
            time.sleep(0.2)
            return
        self._mc.set_gripper_value(int(value), config.GRIPPER_SPEED)
        # set_gripper_value returns immediately; without this the next move can
        # start while the jaws are still closing and drag the cube.
        time.sleep(config.GRIPPER_SETTLE)

    def _send_angles(self, angles, speed: int | None = None) -> None:
        speed = min(int(speed or self.speed), config.MAX_SPEED)
        clamped = [
            max(lo, min(hi, float(v)))
            for v, (lo, hi) in zip(angles, config.JOINT_LIMITS)
        ]
        if self.dry_run:
            log.info("send_angles %s speed=%s", clamped, speed)
            time.sleep(0.5)
            return

        current = self._read(lambda: self._mc.get_angles(), attempts=1)
        if current and all(abs(c - t) <= config.ARRIVAL_TOLERANCE
                           for c, t in zip(current, clamped)):
            log.info("send_angles %s -- already there, skipped", clamped)
            return

        log.info("send_angles %s speed=%s", clamped, speed)
        self._mc.sync_send_angles(clamped, speed, timeout=config.MOVE_TIMEOUT)

    def _send_coords(self, coords, speed: int | None = None) -> None:
        speed = min(int(speed or self.speed), config.MAX_SPEED)
        log.info("send_coords [%s] speed=%s",
                 ", ".join(f"{float(v):.1f}" for v in coords), speed)
        if self.dry_run:
            time.sleep(0.5)
            return
        # mode=1 = linear interpolation, the predictable one for teleop.
        self._mc.send_coords([float(v) for v in coords], speed, 1)
        time.sleep(2.5)     # send_coords is fire-and-forget; give it time to arrive

    def _read_coords(self):
        return self._read(lambda: self._mc.get_coords())

    def _read(self, fn, attempts: int = 3):
        """pymycobot getters return None / -1 / [] when a frame is dropped, which
        happens often enough on USB serial to be worth retrying."""
        if self.dry_run:
            return [0.0] * 6
        for _ in range(attempts):
            value = fn()
            if isinstance(value, list) and len(value) >= 6:
                return value
            time.sleep(0.2)
        return None
