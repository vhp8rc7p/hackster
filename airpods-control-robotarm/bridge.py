#!/usr/bin/env python3
"""
AirPods head pose -> myCobot 280 teleoperation bridge.

Reads UDP JSON pose packets from the `headtrack` Swift helper and maps head
rotation onto arm joints. Motion only happens while the clutch is engaged.

Keys:  SPACE toggle clutch   c recenter   r relax servos   q quit
"""

import argparse
import json
import select
import socket
import sys
import termios
import time
import tty

PORT = "/dev/cu.usbserial-5AE20107941"
BAUD = 115200
UDP_PORT = 9870

# Head axis -> joint index (0-based). myCobot 280 is J1..J6.
#   yaw  (turn head)  -> J1 base rotation
#   pitch(nod head)   -> J2 shoulder
#   roll (tilt head)  -> J5 wrist pitch
AXIS_MAP = [("yaw", 0, 1.0), ("pitch", 1, 1.0), ("roll", 4, 1.0)]

# Conservative soft limits (deg), well inside the myCobot 280 mechanical range.
JOINT_LIMITS = [(-160, 160), (-130, 130), (-145, 145), (-140, 140), (-160, 160), (-175, 175)]

CTRL_HZ = 10.0           # serial link is the bottleneck; 25 Hz floods it
STALE_S = 0.3            # no head packet for this long -> auto-disengage
MAX_STEP_DEG = 3.0       # per control tick, per joint  (=> 30 deg/s at 10 Hz)
MAX_EXCURSION = 45.0     # max deviation from the pose captured at engage


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class HeadSource:
    """Latest head pose from the Swift helper."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.pose = None
        self.stamp = 0.0

    def poll(self):
        """Drain the socket, keeping only the newest packet."""
        while True:
            try:
                data, _ = self.sock.recvfrom(512)
            except BlockingIOError:
                break
            try:
                self.pose = json.loads(data)
                self.stamp = time.time()
            except json.JSONDecodeError:
                pass

    def fresh(self):
        return self.pose is not None and (time.time() - self.stamp) < STALE_S


class Arm:
    def __init__(self, live, speed):
        self.live = live
        self.speed = speed
        self.mc = None
        if live:
            from pymycobot import MyCobot280
            self.mc = MyCobot280(PORT, BAUD, timeout=0.3)
            time.sleep(1.5)
            self.mc.set_fresh_mode(1)   # always act on the newest command

    def angles(self):
        if not self.live:
            return [0.0] * 6
        for _ in range(5):
            a = self.mc.get_angles()
            if a and len(a) == 6:
                return a
            time.sleep(0.1)
        raise RuntimeError("arm not responding to get_angles()")

    def send(self, target):
        if self.live:
            self.mc.send_angles([round(v, 2) for v in target], self.speed)

    def relax(self):
        if self.live:
            self.mc.release_all_servos()


class Teleop:
    def __init__(self, arm, head, gain):
        self.arm, self.head, self.gain = arm, head, gain
        self.engaged = False
        self.head_base = None
        self.arm_base = None
        self.target = None
        self.status = "ready"

    def engage(self):
        if not self.head.fresh():
            self.status = "no head data - is headtrack running?"
            return
        # Capture both baselines so engaging never produces motion, and so
        # accumulated yaw drift is zeroed out on every engage.
        self.head_base = dict(self.head.pose)
        self.arm_base = self.arm.angles()
        self.target = list(self.arm_base)
        self.engaged = True
        self.status = "ENGAGED"

    def disengage(self, why="disengaged"):
        self.engaged = False
        self.status = why

    def step(self):
        if not self.engaged:
            return
        if not self.head.fresh():
            self.disengage("stale head data - disengaged")
            return

        pose = self.head.pose
        desired = list(self.arm_base)
        for axis, joint, scale in AXIS_MAP:
            delta = pose[axis] - self.head_base[axis]
            if axis == "yaw":   # yaw wraps at +/-180
                delta = (delta + 180.0) % 360.0 - 180.0
            delta = clamp(delta * self.gain * scale, -MAX_EXCURSION, MAX_EXCURSION)
            lo, hi = JOINT_LIMITS[joint]
            desired[joint] = clamp(self.arm_base[joint] + delta, lo, hi)

        # Rate limit so a head jerk cannot become an arm jerk.
        for i in range(6):
            step = clamp(desired[i] - self.target[i], -MAX_STEP_DEG, MAX_STEP_DEG)
            self.target[i] += step

        self.arm.send(self.target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually move the arm (default: dry run)")
    ap.add_argument("--gain", type=float, default=1.0, help="deg of joint per deg of head")
    ap.add_argument("--speed", type=int, default=40, help="myCobot speed 1-100")
    ap.add_argument("--auto-engage", action="store_true",
                    help="engage without a keypress (for headless/scripted runs)")
    ap.add_argument("--duration", type=float, default=0.0, help="auto-exit after N seconds")
    args = ap.parse_args()

    head = HeadSource(UDP_PORT)
    arm = Arm(args.live, args.speed)
    tele = Teleop(arm, head, args.gain)

    mode = "LIVE" if args.live else "DRY RUN"
    print(f"\n  AirPods -> myCobot 280   [{mode}]  gain={args.gain}  speed={args.speed}")
    print("  SPACE engage/disengage   c recenter   r relax   q quit\n")

    # Keys need a real terminal; without one (pipe, cron, nohup) run read-only.
    interactive = sys.stdin.isatty()
    fd = sys.stdin.fileno() if interactive else -1
    saved = termios.tcgetattr(fd) if interactive else None
    if not interactive:
        print("  stdin is not a tty - key controls disabled")
        if not args.auto_engage:
            print("  (nothing will move; pass --auto-engage for a scripted run)")

    period = 1.0 / CTRL_HZ
    deadline = time.time() + args.duration if args.duration else None
    try:
        if interactive:
            tty.setcbreak(fd)
        while True:
            if deadline and time.time() > deadline:
                break
            t_tick = time.time()
            head.poll()

            if args.auto_engage and not tele.engaged and head.fresh():
                tele.engage()

            if interactive and select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == " ":
                    tele.disengage() if tele.engaged else tele.engage()
                elif key == "c" and tele.engaged:
                    tele.engage()
                elif key == "r":
                    tele.disengage("servos relaxed")
                    arm.relax()
                elif key in ("q", "\x03"):
                    break

            tele.step()

            pose = head.pose or {}
            live_mark = "*" if head.fresh() else "."
            tgt = " ".join(f"{v:+6.1f}" for v in (tele.target or [0] * 6))
            sys.stdout.write(
                f"\r{live_mark} head p{pose.get('pitch', 0):+6.1f} r{pose.get('roll', 0):+6.1f} "
                f"y{pose.get('yaw', 0):+6.1f} | J {tgt} | {tele.status:<34}"
            )
            sys.stdout.flush()

            time.sleep(max(0.0, period - (time.time() - t_tick)))
    finally:
        if interactive:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        print("\n\nstopped.")


if __name__ == "__main__":
    main()
