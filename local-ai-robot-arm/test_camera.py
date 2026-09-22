#!/usr/bin/env python3
"""
Camera diagnostic for the robot vision pipeline.

Tests the things that actually broke in practice — not just "does it open":

  1. capture      — opens, resolution, real FPS, dropped/frozen frames
  2. exposure     — is auto-exposure DRIFTING? (this is why a cube would be
                    detected, then vanish when you moved your hand)
  3. sharpness    — is the image in focus enough to see a 25mm cube?
  4. colour       — do the HSV ranges in qwen_command actually match your
                    cubes under the CURRENT lighting?
  5. markers      — are the ArUco calibration markers still detectable?
  6. workspace    — how much of the frame maps to reachable robot coords?
  7. latency      — how stale is a captured frame?

USAGE
  ./mlx_env/bin/python test_camera.py                 # all tests
  ./mlx_env/bin/python test_camera.py --seconds 10    # longer exposure watch
  ./mlx_env/bin/python test_camera.py --show          # save annotated PNG
  ./mlx_env/bin/python test_camera.py --only colour   # one test

Exit code 0 = all pass, 1 = something failed.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
CALIB_PATH = "calibration_affine2d.json"
ARUCO_DICT = cv2.aruco.DICT_6X6_50

# Must match qwen_command.CUBE_HSV_RANGES / test_cube_hover.COLOR_RANGES
COLOR_RANGES = {
    "green":  [(35, 80, 60), (85, 255, 255)],
    "blue":   [(95, 80, 60), (135, 255, 255)],
    "pink":   [(140, 60, 100), (175, 255, 255)],
    "yellow": [(18, 80, 80), (35, 255, 255)],
    "red":    [(0, 100, 80), (10, 255, 255)],
}
MIN_CUBE_AREA_PX = 500

_results = []


def check(name, ok, detail="", warn=False):
    _results.append((name, bool(ok), detail, warn))
    mark = "✓" if ok else ("⚠" if warn else "✗")
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def section(t):
    print(f"\n── {t} " + "─" * max(0, 52 - len(t)))


def grab(cap, n=1):
    """Read n frames, return the last (flushes stale buffer)."""
    f = None
    for _ in range(n):
        ret, fr = cap.read()
        if ret:
            f = fr
    return f


# ── 1. capture ────────────────────────────────────────────────────────
def test_capture(cap):
    section("capture")
    f = grab(cap, 5)
    if not check("returns a frame", f is not None):
        return None
    h, w = f.shape[:2]
    check("resolution matches config", (w, h) == (FRAME_W, FRAME_H),
          f"got {w}x{h}, configured {FRAME_W}x{FRAME_H}")

    # Real achievable FPS
    t0, n = time.time(), 20
    for _ in range(n):
        cap.read()
    fps = n / (time.time() - t0)
    check("capture rate >= 10 fps", fps >= 10, f"{fps:.1f} fps")

    # Frozen-feed detection: consecutive frames should differ at least a little
    a = grab(cap, 2).astype(np.int16)
    time.sleep(0.15)
    b = grab(cap, 2).astype(np.int16)
    diff = float(np.mean(np.abs(a - b)))
    check("feed is live (frames change)", diff > 0.05,
          f"mean frame-to-frame delta {diff:.3f}")
    return f


# ── 2. exposure stability (the colour-drift bug) ──────────────────────
def test_exposure(cap, seconds):
    section(f"exposure / white-balance stability ({seconds:.0f}s watch)")
    brightness, wb = [], []
    t0 = time.time()
    while time.time() - t0 < seconds:
        f = grab(cap, 1)
        if f is None:
            continue
        brightness.append(float(f.mean()))
        b, g, r = [float(f[:, :, i].mean()) for i in range(3)]
        wb.append(r / max(b, 1e-6))          # red/blue ratio ≈ white balance
        time.sleep(0.1)

    if not brightness:
        return check("collected exposure samples", False)

    bmean, bmin, bmax = np.mean(brightness), min(brightness), max(brightness)
    swing = (bmax - bmin) / max(bmean, 1e-6) * 100
    check("brightness in a usable range", 25 < bmean < 230, f"mean {bmean:.0f}/255")
    check("exposure is STABLE (<8% swing)", swing < 8.0,
          f"{swing:.1f}% swing ({bmin:.0f}–{bmax:.0f})", warn=True)
    if swing >= 8.0:
        print("      ↳ auto-exposure is drifting: a cube can fall OUT of its HSV")
        print("        range when the scene changes. Lock exposure, or re-sample")
        print("        the colour (click-to-sample in test_cube_hover.py).")

    wswing = (max(wb) - min(wb)) / max(np.mean(wb), 1e-6) * 100
    check("white balance is STABLE (<6% swing)", wswing < 6.0,
          f"{wswing:.1f}% swing", warn=True)


# ── 3. focus ──────────────────────────────────────────────────────────
def test_sharpness(cap):
    section("focus / sharpness")
    f = grab(cap, 3)
    if f is None:
        return
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    # Laplacian variance: <50 is visibly blurry for this kind of scene
    check("image is in focus", lap > 50, f"laplacian variance {lap:.0f}")
    if lap <= 50:
        print("      ↳ blurry: a 25mm cube may not be detectable. Check focus ring")
        print("        / clean the lens / make sure the gantry isn't vibrating.")


# ── 4. colour ranges vs real cubes ────────────────────────────────────
def test_colour(cap, save=False):
    section("cube colour detection (current lighting)")
    f = grab(cap, 3)
    if f is None:
        return
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    found_any = False
    disp = f.copy()
    for name, (lo, hi) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in cnts if cv2.contourArea(c) >= MIN_CUBE_AREA_PX]
        if big:
            found_any = True
            c = max(big, key=cv2.contourArea)
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
            print(f"  ✓ {name:<7} found  area={int(area):>6}px  at ({cx:.0f},{cy:.0f})")
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(disp, name, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        else:
            px = int(mask.sum() / 255)
            print(f"  · {name:<7} not found ({px} matching px, need a "
                  f"{MIN_CUBE_AREA_PX}px blob)")
    check("at least one cube colour is detectable", found_any,
          "" if found_any else "no cube on the table, or HSV ranges are wrong "
                               "for this lighting")
    if save:
        cv2.imwrite("camera_check.png", disp)
        print("      ↳ saved camera_check.png")


# ── 5. aruco markers ──────────────────────────────────────────────────
def test_markers(cap):
    section("ArUco calibration markers")
    f = grab(cap, 3)
    if f is None:
        return
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    corners, ids, _ = cv2.aruco.ArucoDetector(
        d, cv2.aruco.DetectorParameters()).detectMarkers(f)
    n = 0 if ids is None else len(ids)
    if n:
        print(f"      visible IDs: {sorted(int(i) for i in ids.flatten())}")
    check("markers visible (optional — only needed to re-calibrate)",
          n > 0, f"{n} detected", warn=True)


# ── 6. workspace coverage ─────────────────────────────────────────────
def test_workspace():
    section("workspace coverage (frame → reachable robot coords)")
    if not os.path.exists(CALIB_PATH):
        check(f"{CALIB_PATH} present", False, "run affine2d_calibrate_multi.py")
        return
    with open(CALIB_PATH) as fh:
        d = json.load(fh)
    A = np.array(d["affine_2x3"], dtype=float)
    w, h = d.get("image_size", [FRAME_W, FRAME_H])

    MAX_REACH, LIMIT, SEAL_Z = 280.0, 281.45, 55.0
    usable = grid = 0
    for u in np.linspace(0, w, 20):
        for v in np.linspace(0, h, 20):
            x, y = A @ np.array([u, v, 1.0])
            grid += 1
            if (abs(x) <= LIMIT and abs(y) <= LIMIT and
                    np.sqrt(x*x + y*y + SEAL_Z**2) <= MAX_REACH):
                usable += 1
    frac = usable / grid
    check("enough of the frame is reachable", frac >= 0.15,
          f"{frac*100:.0f}% of the frame maps inside the arm's reach")
    if frac < 0.35:
        print("      ↳ most of what the camera sees is out of reach. That's fine")
        print("        if cubes sit in the reachable zone, but detections outside")
        print("        it are now refused (they used to crash the program).")


# ── 7. latency ────────────────────────────────────────────────────────
def test_latency(cap):
    section("frame latency / staleness")
    t0 = time.time()
    grab(cap, 1)
    one = (time.time() - t0) * 1000
    t0 = time.time()
    grab(cap, 3)
    three = (time.time() - t0) * 1000
    print(f"      1 read {one:.0f}ms   3 reads {three:.0f}ms")
    check("single read is fast", one < 200, f"{one:.0f}ms")


def main():
    ap = argparse.ArgumentParser(description="Camera diagnostic")
    ap.add_argument("--camera", type=int, default=CAMERA_ID)
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="how long to watch exposure drift")
    ap.add_argument("--show", action="store_true", help="save camera_check.png")
    ap.add_argument("--only", default=None,
                    choices=["capture", "exposure", "sharpness", "colour",
                             "markers", "workspace", "latency"])
    args = ap.parse_args()

    print("=" * 60)
    print("  CAMERA DIAGNOSTIC")
    print("=" * 60)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8):
        cap.read()
    if not cap.isOpened():
        print(f"  ✗ could not open camera {args.camera}")
        return 1

    only = args.only
    try:
        if only in (None, "capture"):
            test_capture(cap)
        if only in (None, "exposure"):
            test_exposure(cap, args.seconds)
        if only in (None, "sharpness"):
            test_sharpness(cap)
        if only in (None, "colour"):
            test_colour(cap, save=args.show)
        if only in (None, "markers"):
            test_markers(cap)
        if only in (None, "workspace"):
            test_workspace()
        if only in (None, "latency"):
            test_latency(cap)
    finally:
        cap.release()

    hard = [(n, d) for n, ok, d, warn in _results if not ok and not warn]
    warns = [(n, d) for n, ok, d, warn in _results if not ok and warn]
    passed = sum(1 for _, ok, _, _ in _results if ok)
    print("\n" + "=" * 60)
    print(f"  {passed}/{len(_results)} passed"
          + (f", {len(warns)} warning(s)" if warns else ""))
    for n, d in warns:
        print(f"    ⚠ {n}  — {d}")
    for n, d in hard:
        print(f"    ✗ {n}  — {d}")
    print("  " + ("ALL GOOD" if not hard else "FAILURES ABOVE"))
    print("=" * 60)
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
