#!/usr/bin/env python3
"""
Black-box test suite for the robot pipeline.

Catches broken config / math / logic BEFORE you run the arm — most tests need
no robot and no camera, so you can run them any time in a few seconds.

This exists because bugs kept reaching the hardware: an out-of-range target
crashed the program mid-demo, hover heights were silently unreachable, and a
misheard "green queue" sent OWL hunting for nothing. Each of those is now a
test.

USAGE
  ./mlx_env/bin/python test_system.py             # offline tests only (fast)
  ./mlx_env/bin/python test_system.py --robot     # + robot connection tests
  ./mlx_env/bin/python test_system.py --camera    # + camera tests
  ./mlx_env/bin/python test_system.py --all       # everything

Exit code is 0 if everything passed, 1 otherwise — so it works in a script.
"""

import argparse
import json
import os
import sys

import numpy as np

# ── tiny test harness (no pytest dependency) ──────────────────────────
_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    mark = "✓" if cond else "✗"
    line = f"  {mark} {name}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return bool(cond)


def section(title):
    print(f"\n── {title} " + "─" * max(0, 56 - len(title)))


def summary():
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [(n, d) for n, ok, d in _results if not ok]
    print("\n" + "=" * 64)
    print(f"  {passed}/{len(_results)} passed")
    if failed:
        print(f"\n  FAILURES:")
        for n, d in failed:
            print(f"    ✗ {n}" + (f"  — {d}" if d else ""))
        print("=" * 64)
        return 1
    print("  ALL PASS")
    print("=" * 64)
    return 0


# ── offline tests ─────────────────────────────────────────────────────
def test_config(Q):
    """Geometry constants must be self-consistent AND physically reachable."""
    section("config / geometry")

    table = Q.TABLE_Z_BASE_MM
    cube_top = table + Q.CUBE_HEIGHT_MM
    seal_z = cube_top + Q.TOUCH_ABOVE
    pick_hover_z = cube_top + Q.PICK_HOVER_ABOVE
    place_hover_z = cube_top + Q.HOVER_ABOVE

    check("table height is plausible", 0 <= table <= 200, f"{table}mm")
    check("seal Z is above the table", seal_z > table,
          f"seal {seal_z}mm vs table {table}mm")
    check("seal Z is below the cube top (presses in)", seal_z <= cube_top,
          f"seal {seal_z}mm, cube top {cube_top}mm")
    check("pick hover is above the seal", pick_hover_z > seal_z,
          f"hover {pick_hover_z}mm vs seal {seal_z}mm")

    # The bug that bit us: a hover height that's unreachable at workspace edge.
    # Check the hover is reachable at a realistic far-corner XY.
    far_xy = 230.0   # a cube out near the edge of the working area
    for label, z in (("pick hover", pick_hover_z), ("place hover", place_hover_z)):
        reach = float(np.sqrt(far_xy ** 2 + z ** 2))
        check(f"{label} reachable at far XY ({far_xy:.0f}mm out)",
              reach <= Q.MAX_REACH_MM,
              f"needs {reach:.0f}mm, max {Q.MAX_REACH_MM:.0f}mm")

    check("seal Z clears the tip floor", seal_z >= Q.MIN_TIP_Z_MM,
          f"seal {seal_z}mm, floor {Q.MIN_TIP_Z_MM}mm")
    check("pump length set", 0 < Q.PUMP_LENGTH < 200, f"{Q.PUMP_LENGTH}mm")


def test_validate_target(Q):
    """The guard that stops pymycobot from raising and killing the program."""
    section("target validation (crash guard)")
    v = Q.validate_target

    ok, _ = v(150, 50, 80)
    check("accepts a normal in-range target", ok)

    # This is the exact value that crashed a live run.
    ok, why = v(171.9, -404.2, 45.0)
    check("rejects the y=-404mm target that crashed the demo", not ok, why)

    ok, _ = v(300, 0, 80)
    check("rejects x beyond the firmware limit", not ok)
    ok, _ = v(0, 0, 400)
    check("rejects z beyond the firmware limit", not ok)
    ok, _ = v(250, 250, 100)
    check("rejects targets past the arm's reach", not ok)
    ok, _ = v(150, 50, -10)
    check("rejects a target below the table floor", not ok)
    ok, _ = v(float("nan"), 50, 80)
    check("rejects NaN coordinates", not ok)
    ok, _ = v(float("inf"), 50, 80)
    check("rejects infinite coordinates", not ok)


def test_normalize(Q):
    """Misheard object names must still resolve to the right target."""
    section("speech normalization")
    n = Q.normalize_object

    for heard in ("a green queue", "a green cue", "a green cave", "a green cube"):
        check(f"'{heard}' → green cube", n(heard) == "a green cube", n(heard))
    check("'a blue queue' → blue cube", n("a blue queue") == "a blue cube")

    # Must NOT rewrite non-cube targets.
    for keep in ("a human hand", "a cardboard box", "a credit card"):
        check(f"leaves '{keep}' alone", n(keep) == keep, n(keep))
    check("handles None", n(None) is None)


def test_fast_command(Q):
    """Short commands the STT mangles must still work (or safely do nothing)."""
    section("fast-path commands")
    f = Q._fast_command

    for s in ("home", "Home.", "go home", "reset", "Homing."):
        check(f"'{s}' → home", f(s) == "home", str(f(s)))
    for s in ("quit", "Exit."):
        check(f"'{s}' → quit", f(s) == "quit", str(f(s)))
    # Must not hijack real commands, and must not fire on noise.
    for s in ("pick the green cube", "place it in the box", "Oh.", "Ho."):
        check(f"'{s}' does not fast-path", f(s) is None, str(f(s)))


def test_interrupt_words(Q):
    section("interrupt / release words")
    check("'stop' is an interrupt", Q._is_interrupt_text("stop"))
    check("'cancel' is an interrupt", Q._is_interrupt_text("cancel"))
    check("'wait' is an interrupt", Q._is_interrupt_text("Wait."))
    check("'drop' is a release", Q._is_release_text("drop it"))
    check("a normal command is not an interrupt",
          not Q._is_interrupt_text("pick the green cube"))


def test_calibration(Q):
    """The affine calibration must exist, be well-fitted, and map the visible
    frame to coordinates the arm can actually reach."""
    section("calibration")
    path = Q.AFFINE_CALIB_PATH
    if not os.path.exists(path):
        check(f"{path} exists", False, "run affine2d_calibrate_multi.py")
        return
    check(f"{path} exists", True)

    with open(path) as f:
        d = json.load(f)

    A = np.array(d.get("affine_2x3", []), dtype=float)
    check("affine matrix is 2x3", A.shape == (2, 3), str(A.shape))

    res = d.get("residuals_mm", [])
    if res:
        mean_r, max_r = float(np.mean(res)), float(np.max(res))
        check("mean residual < 5mm", mean_r < 5.0, f"{mean_r:.2f}mm")
        check("max residual < 10mm", max_r < 10.0, f"{max_r:.2f}mm")
    n = d.get("n_markers", 0)
    check("fitted from >= 4 markers", n >= 4, f"{n} markers")

    # Sanity: the center of the image should map somewhere reachable.
    if A.shape == (2, 3):
        w, h = d.get("image_size", [Q.FRAME_W, Q.FRAME_H])
        cx, cy = w / 2.0, h / 2.0
        x, y = A @ np.array([cx, cy, 1.0])
        seal_z = Q.TABLE_Z_BASE_MM + Q.CUBE_HEIGHT_MM + Q.TOUCH_ABOVE
        ok, why = Q.validate_target(float(x), float(y), seal_z)
        check("image center maps to a reachable pick",
              ok, f"({x:.0f}, {y:.0f}) → {why}")

        # How much of the frame is actually usable? Pure information, but a
        # very low number means the camera is aimed badly.
        usable = 0
        total = 0
        for u in np.linspace(0, w, 12):
            for vv in np.linspace(0, h, 12):
                px, py = A @ np.array([u, vv, 1.0])
                total += 1
                if Q.validate_target(float(px), float(py), seal_z)[0]:
                    usable += 1
        frac = usable / total
        check("at least 15% of the frame is reachable",
              frac >= 0.15, f"{frac*100:.0f}% of frame reachable")


def test_files():
    section("required files")
    for f in ("qwen_command.py", "mycobot_280_m5.urdf", "calibration_affine2d.json"):
        check(f"{f} present", os.path.exists(f))
    check("tts/ prompts present",
          os.path.isdir("tts") and len(os.listdir("tts")) > 0)


# ── hardware tests (opt-in) ───────────────────────────────────────────
def test_robot(Q):
    section("robot (live)")
    import glob
    ports = sorted(glob.glob("/dev/tty.usbserial-*"))
    if not check("serial port present", bool(ports), str(ports)):
        return
    port_ok = Q.SERIAL_PORT in ports
    check("configured SERIAL_PORT matches a real port", port_ok,
          f"configured {Q.SERIAL_PORT}")
    if not port_ok:
        return
    try:
        from pymycobot.mycobot280 import MyCobot280
        import time
        mc = MyCobot280(Q.SERIAL_PORT, Q.BAUD_RATE)
        time.sleep(1.5)
        angles = mc.get_angles()
        check("reads joint angles", isinstance(angles, (list, tuple)) and len(angles) == 6,
              str(angles))
        coords = mc.get_coords()
        check("reads TCP coords", isinstance(coords, (list, tuple)) and len(coords) == 6,
              str(coords))
        if Q.IK_BACKEND == "native_tool":
            check("tool-frame API available", hasattr(mc, "set_tool_reference"))
    except Exception as e:
        check("robot connection", False, str(e))


def test_camera(Q):
    section("camera (live)")
    try:
        import cv2
        cap = cv2.VideoCapture(Q.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Q.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Q.FRAME_H)
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if check("camera opens and returns a frame", ret and frame is not None):
            h, w = frame.shape[:2]
            check("frame matches configured resolution",
                  (w, h) == (Q.FRAME_W, Q.FRAME_H), f"got {w}x{h}")
            check("frame is not black",
                  float(frame.mean()) > 5.0, f"mean {frame.mean():.1f}")
        cap.release()
    except Exception as e:
        check("camera", False, str(e))


def main():
    ap = argparse.ArgumentParser(description="Black-box tests for the pipeline")
    ap.add_argument("--robot", action="store_true",
                    help="ADD robot tests to the offline ones")
    ap.add_argument("--camera", action="store_true",
                    help="ADD camera tests to the offline ones "
                         "(for a full camera diagnostic use test_camera.py)")
    ap.add_argument("--all", action="store_true", help="test everything")
    ap.add_argument("--only", action="store_true",
                    help="with --robot/--camera, run ONLY those, skipping the "
                         "offline tests")
    args = ap.parse_args()

    print("=" * 64)
    print("  ROBOT PIPELINE — BLACK-BOX TESTS")
    print("=" * 64)
    print("  importing qwen_command (loads torch/mlx — takes a few seconds)...")
    try:
        import qwen_command as Q
    except Exception as e:
        print(f"\n  ✗ could not import qwen_command: {e}")
        return 1
    print(f"  ok. IK_BACKEND={Q.IK_BACKEND}  CALIB_MODE={Q.CALIB_MODE}  "
          f"INPUT_MODE={Q.INPUT_MODE}")

    hw_only = args.only and (args.robot or args.camera)
    if not hw_only:
        test_files()
        test_config(Q)
        test_validate_target(Q)
        test_normalize(Q)
        test_fast_command(Q)
        test_interrupt_words(Q)
        test_calibration(Q)

    if args.robot or args.all:
        test_robot(Q)
    if args.camera or args.all:
        test_camera(Q)

    return summary()


if __name__ == "__main__":
    sys.exit(main())
