#!/usr/bin/env python3
"""
Measure the ARM's positioning error — commanded vs where it actually goes.

The vision side is already proven accurate (qwen and the hover test agree to
0.6mm). So if a pick lands 1-2cm off, the error is in the motion. This
isolates it by replaying BOTH motion patterns to the same cube:

  A) HOVER-TEST pattern : ONE move, straight to tip Z=80  (it only hovers)
  B) QWEN pattern       : TWO moves, approach Z=110 then descend to Z=55

After each move it reads get_coords() back and reports the error. That tells
us which of two things is true:

  * error is bigger at Z=55 than Z=80  -> depth-dependent arm error
  * error is the same at both          -> the two-move sequence is the cause

No pump is fired. The cube is located by HSV (fast, no OWL needed) — the
coordinate is identical to what qwen computes, which we verified separately.

USAGE
  ./mlx_env/bin/python test_motion_accuracy.py                  # both patterns
  ./mlx_env/bin/python test_motion_accuracy.py --pattern qwen   # just B
  ./mlx_env/bin/python test_motion_accuracy.py --repeat 3       # repeatability
  ./mlx_env/bin/python test_motion_accuracy.py --color red
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np
from pymycobot.mycobot280 import MyCobot280

SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
CALIB_PATH = "calibration_affine2d.json"

PUMP_LENGTH = 70.0
SPEED = 30
HOME_ANGLES = [0, 0, 0, 0, 0, 0]
DOWN_ORIENTATION = (180.0, 0.0, 0.0)

# Heights (PUMP TIP, tool frame active) — must match the two scripts.
HOVER_TEST_Z = 80.0      # test_cube_hover.py HOVER_Z_MM
QWEN_APPROACH_Z = 110.0  # TABLE(35)+CUBE(25)+PICK_HOVER_ABOVE(50)
QWEN_DOWN_Z = 55.0       # TABLE(35)+CUBE(25)+TOUCH_ABOVE(-5)

COLOR_RANGES = {
    "green":  [(35, 80, 60), (85, 255, 255)],
    "blue":   [(95, 80, 60), (135, 255, 255)],
    "pink":   [(140, 60, 100), (175, 255, 255)],
    "yellow": [(18, 80, 80), (35, 255, 255)],
    "red":    [(0, 100, 80), (10, 255, 255)],
}


def find_cube(cap, color):
    """HSV centroid of the largest matching blob → (u, v) pixel."""
    for _ in range(4):
        cap.grab()
    ret, f = cap.read()
    if not ret:
        return None
    lo, hi = COLOR_RANGES[color]
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 500]
    if not cnts:
        return None
    M = cv2.moments(max(cnts, key=cv2.contourArea))
    if M["m00"] == 0:
        return None
    return (M["m10"]/M["m00"], M["m01"]/M["m00"])


def settle(mc, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if mc.is_moving() == 0:
                break
        except Exception:
            pass
        time.sleep(0.25)
    time.sleep(0.5)


def move_and_measure(mc, x, y, z, label):
    """Send one coord move, wait, read back. Returns (dx, dy, dz, dxy, actual)."""
    rx, ry, rz = DOWN_ORIENTATION
    target = [float(x), float(y), float(z), rx, ry, rz]
    try:
        mc.send_coords(target, SPEED, 0)
    except Exception as e:
        print(f"    {label}: REFUSED by firmware — {e}")
        return None
    settle(mc)
    a = mc.get_coords()
    if not isinstance(a, (list, tuple)) or len(a) != 6:
        print(f"    {label}: could not read back pose")
        return None
    dx, dy, dz = a[0]-x, a[1]-y, a[2]-z
    dxy = float(np.hypot(dx, dy))
    print(f"    {label:<22} cmd=({x:6.1f},{y:6.1f},{z:5.1f})  "
          f"act=({a[0]:6.1f},{a[1]:6.1f},{a[2]:5.1f})  "
          f"dXY={dxy:5.1f}mm  dZ={dz:+5.1f}mm")
    return dx, dy, dz, dxy, a


def main():
    ap = argparse.ArgumentParser(description="Arm positioning accuracy test")
    ap.add_argument("--color", default="green", choices=list(COLOR_RANGES))
    ap.add_argument("--pattern", default="both",
                    choices=["both", "hover", "qwen"])
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat each pattern N times (shows repeatability)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    A = np.array(json.load(open(CALIB_PATH))["affine_2x3"], dtype=float)

    print("=" * 74)
    print("  ARM POSITIONING ACCURACY — commanded vs actual")
    print("=" * 74)
    print(f"  A) hover-test pattern : 1 move  -> tip Z={HOVER_TEST_Z:.0f}")
    print(f"  B) qwen pattern       : 2 moves -> Z={QWEN_APPROACH_Z:.0f} "
          f"then Z={QWEN_DOWN_Z:.0f}")
    print(f"  colour: {args.color}   repeats: {args.repeat}   NO PUMP is fired")
    print("=" * 74)
    if not args.yes:
        if input("\n  THE ARM WILL MOVE and touch the cube. Continue? [y/N] "
                 ).strip().lower() not in ("y", "yes"):
            return 0

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8):
        cap.read()

    uv = find_cube(cap, args.color)
    if uv is None:
        cap.release()
        sys.exit(f"No {args.color} cube found — put one on the table.")
    x, y = A @ np.array([uv[0], uv[1], 1.0])
    print(f"\n  cube at pixel ({uv[0]:.0f}, {uv[1]:.0f}) -> base XY "
          f"({x:.1f}, {y:.1f})")

    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    try:
        mc.power_on(); time.sleep(0.5)
    except Exception:
        pass
    try:
        mc.set_tool_reference([0, 0, PUMP_LENGTH, 0, 0, 0])
        mc.set_end_type(1)
        print("  tool frame set (send_coords drives the PUMP TIP)")
    except Exception as e:
        print(f"  ⚠ tool frame failed: {e}")

    res = {"hover": [], "qwen_approach": [], "qwen_down": []}
    try:
        for rep in range(args.repeat):
            print(f"\n  ── repeat {rep+1}/{args.repeat} " + "─" * 40)

            if args.pattern in ("both", "hover"):
                print("  A) HOVER-TEST pattern (single move)")
                mc.send_angles(HOME_ANGLES, SPEED); settle(mc)
                r = move_and_measure(mc, x, y, HOVER_TEST_Z, "hover Z=80")
                if r:
                    res["hover"].append(r[3])

            if args.pattern in ("both", "qwen"):
                print("  B) QWEN pattern (approach, then descend)")
                mc.send_angles(HOME_ANGLES, SPEED); settle(mc)
                r1 = move_and_measure(mc, x, y, QWEN_APPROACH_Z, "approach Z=110")
                if r1:
                    res["qwen_approach"].append(r1[3])
                r2 = move_and_measure(mc, x, y, QWEN_DOWN_Z, "down Z=55")
                if r2:
                    res["qwen_down"].append(r2[3])
                move_and_measure(mc, x, y, QWEN_APPROACH_Z, "lift back Z=110")
    finally:
        print("\n  returning home...")
        try:
            mc.send_angles(HOME_ANGLES, SPEED); settle(mc)
            mc.set_end_type(0)
        except Exception:
            pass
        cap.release()

    print("\n" + "=" * 74)
    print("  RESULTS  (XY error, mm)")
    print("=" * 74)
    for k, label in (("hover", "A) hover-test  Z=80 (1 move)"),
                     ("qwen_approach", "B) qwen approach Z=110"),
                     ("qwen_down", "B) qwen down     Z=55  ← the pick height")):
        v = res[k]
        if v:
            print(f"  {label:<38} mean {np.mean(v):5.1f}   "
                  f"min {min(v):5.1f}   max {max(v):5.1f}   n={len(v)}")

    h, d = res["hover"], res["qwen_down"]
    if h and d:
        print(f"\n  hover Z=80 error : {np.mean(h):.1f} mm")
        print(f"  qwen  Z=55 error : {np.mean(d):.1f} mm")
        diff = np.mean(d) - np.mean(h)
        print(f"  difference       : {diff:+.1f} mm")
        if diff > 3:
            print("\n  → Error is LARGER at the lower pick height. The arm loses")
            print("    accuracy as it extends down. A Z-dependent XY offset, or")
            print("    picking with a less-extended pose, would help.")
        elif diff < -3:
            print("\n  → Error is larger at the HIGHER pose — unexpected; likely")
            print("    the two-move sequence, not depth.")
        else:
            print("\n  → Both heights are about equal, so depth is NOT the cause.")
            print("    The difference you saw comes from the approach sequence")
            print("    or from parallax at that cube position.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
