#!/usr/bin/env python3
"""
2D affine calibration using a COLOURED CUBE instead of ArUco markers.

Same output as affine2d_calibrate_multi.py (calibration_affine2d.json), but you
only need one cube — move it around the workspace instead of printing markers.

WHY THIS CAN BE BETTER THAN ARUCO HERE:
  The ArUco version calibrates the TABLE plane (markers lie flat), but you
  detect the TOP of a 40mm cube when picking. That height difference projects
  outward from the camera and costs a few mm, growing with distance from the
  image centre. Calibrating with the cube itself, at cube height, folds that
  error into the fit — so cube picks land better. (The trade-off: it's then
  slightly off for FLAT objects like cards.)

WORKFLOW (repeat for 6-8 spots spread across the reachable area):
  1. Put the cube somewhere, keep the arm out of the camera's view
  2. SPACE  — locks the cube's pixel position
  3. Drag the arm by hand so the pump tip touches the CENTRE of the cube top
  4. SPACE  — records the arm position
  5. Move the cube and repeat. ENTER when done -> solves and saves.

Keys:  SPACE = lock / record    S = skip this point    U = undo last    Q = solve & quit

USAGE
  ./mlx_env/bin/python affine2d_calibrate_cube.py
  ./mlx_env/bin/python affine2d_calibrate_cube.py --color red
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from pymycobot.mycobot280 import MyCobot280

SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
PUMP_LENGTH = 70.0
OUT_PATH = "/Users/v/local-ai-robot-arm/calibration_affine2d.json"
MIN_POINTS = 4
MIN_AREA_PX = 500

COLOR_RANGES = {
    "green":  [((30, 70, 50),  (90, 255, 255))],
    "blue":   [((95, 80, 60),  (135, 255, 255))],
    "pink":   [((140, 60, 100), (172, 255, 255))],
    "yellow": [((18, 80, 80),  (35, 255, 255))],
    "red":    [((0, 90, 70),   (10, 255, 255)),
               ((165, 50, 70), (179, 255, 255))],
}


def bands_mask(hsv, bands):
    m = None
    for lo, hi in bands:
        lo_a = np.array([int(round(v)) for v in lo], dtype=np.uint8)
        hi_a = np.array([int(round(v)) for v in hi], dtype=np.uint8)
        mm = cv2.inRange(hsv, lo_a, hi_a)
        m = mm if m is None else cv2.bitwise_or(m, mm)
    return m


def detect_cube(frame, bands):
    """Return ((cx, cy), area, bbox) for the largest matching blob, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = bands_mask(hsv, bands)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= MIN_AREA_PX]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    x, y, w, h = cv2.boundingRect(c)
    return ((M["m10"] / M["m00"], M["m01"] / M["m00"]),
            cv2.contourArea(c), (x, y, x + w, y + h))


def coords_to_T(coords):
    """[x,y,z,rx,ry,rz] (mm/deg) -> 4x4 transform, xyz extrinsic convention."""
    x, y, z, rx, ry, rz = coords
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def read_tip(mc, n=8):
    """Average the pump-tip position in base coords over n reads."""
    tcps = []
    for _ in range(n):
        try:
            c = mc.get_coords()
        except Exception:
            c = None
        if isinstance(c, (list, tuple)) and len(c) == 6:
            tcps.append(c)
        time.sleep(0.08)
    if len(tcps) < 3:
        return None
    tcp = np.mean(tcps, axis=0)
    T = coords_to_T(tcp)
    return (T @ np.array([0, 0, PUMP_LENGTH, 1.0]))[:3]


def main():
    ap = argparse.ArgumentParser(description="2D affine calibration with a cube")
    ap.add_argument("--color", default="green", choices=list(COLOR_RANGES))
    args = ap.parse_args()
    bands = COLOR_RANGES[args.color]

    print("=" * 70)
    print("  2D AFFINE CALIBRATION USING A CUBE")
    print("=" * 70)
    print(f"  cube colour: {args.color}")
    print(f"  Spread {MIN_POINTS}-8 positions across the REACHABLE area.")
    print("  SPACE = lock pixel / record touch   S = skip   U = undo   Q = solve")
    print("=" * 70)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8):
        cap.read()

    print("\nConnecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    try:
        mc.power_on(); time.sleep(0.5)
    except Exception:
        pass
    print("Releasing servos so you can move the arm by hand...")
    for sid in range(1, 7):
        try: mc.release_servo(sid)
        except Exception: pass
    time.sleep(1)
    print("Arm should now be limp.\n")

    win = "cube_affine_calib"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    pairs = []          # (pixel_xy, base_xy)
    locked_px = None    # pixel locked, waiting for the touch

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        det = detect_cube(frame, bands)
        disp = frame.copy()

        # already-collected points
        for i, (px, _) in enumerate(pairs):
            cv2.circle(disp, (int(px[0]), int(px[1])), 10, (0, 200, 0), 2)
            cv2.putText(disp, str(i + 1), (int(px[0]) + 12, int(px[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        if locked_px is not None:
            cv2.drawMarker(disp, (int(locked_px[0]), int(locked_px[1])),
                           (0, 0, 255), cv2.MARKER_CROSS, 44, 3)
            cv2.circle(disp, (int(locked_px[0]), int(locked_px[1])), 26, (0, 0, 255), 3)
            msg = "TOUCH the cube top with the pump tip, then SPACE"
            col = (0, 0, 255)
        elif det is not None:
            (cx, cy), area, (x1, y1, x2, y2) = det
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(disp, (int(cx), int(cy)), 6, (0, 0, 255), -1)
            msg = "cube found - SPACE to lock this position"
            col = (0, 255, 255)
        else:
            msg = f"no {args.color} cube visible"
            col = (0, 0, 255)

        cv2.putText(disp, f"[{len(pairs)} points]  {msg}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(disp, "SPACE=lock/record   S=skip   U=undo   Q=solve & quit",
                    (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord('u') and pairs:
            pairs.pop(); locked_px = None
            print(f"  undo — {len(pairs)} points left")
            continue
        if k == ord('s'):
            locked_px = None
            continue
        if k == 32:
            if locked_px is None:
                if det is None:
                    print("  no cube detected — move it into view")
                    continue
                locked_px = det[0]
                print(f"  locked pixel ({locked_px[0]:.0f}, {locked_px[1]:.0f}) — "
                      f"now touch the cube top with the pump tip")
            else:
                tip = read_tip(mc)
                if tip is None:
                    print("  couldn't read arm pose — try again")
                    continue
                pairs.append((locked_px, tip[:2]))
                print(f"  ✓ point {len(pairs)}: pixel "
                      f"({locked_px[0]:.0f},{locked_px[1]:.0f}) -> base "
                      f"({tip[0]:.1f},{tip[1]:.1f})")
                locked_px = None

    cv2.destroyAllWindows()
    for _ in range(6):
        cv2.waitKey(1)
    cap.release()

    if len(pairs) < MIN_POINTS:
        print(f"\n✗ only {len(pairs)} points, need >= {MIN_POINTS}. Nothing saved.")
        return

    pixels = np.array([p[0] for p in pairs], dtype=np.float32)
    base = np.array([p[1] for p in pairs], dtype=np.float32)
    P = np.hstack([pixels, np.ones((len(pixels), 1), dtype=np.float32)])
    A_T, _, _, _ = np.linalg.lstsq(P, base, rcond=None)
    A = A_T.T.astype(np.float32)

    pred = (A @ P.T).T
    res = np.linalg.norm(pred - base, axis=1)
    print("\n── Solve ──")
    print(f"  fit from {len(pairs)} cube positions")
    print(f"  A = [{A[0,0]:+.6f} {A[0,1]:+.6f} {A[0,2]:+.2f}]")
    print(f"      [{A[1,0]:+.6f} {A[1,1]:+.6f} {A[1,2]:+.2f}]")
    for i, r in enumerate(res, 1):
        print(f"    point {i}: residual {r:.2f} mm")
    print(f"  mean {res.mean():.2f} mm   max {res.max():.2f} mm")
    if res.mean() > 5:
        print("  ⚠ mean residual > 5mm — touches may have been imprecise, or the")
        print("    cube moved between locking the pixel and touching it.")

    if os.path.exists(OUT_PATH):
        bak = OUT_PATH + ".bak"
        os.replace(OUT_PATH, bak)
        print(f"  (previous calibration backed up to {os.path.basename(bak)})")

    out = {
        "mode": "affine2d_pixel_to_base_xy_cube",
        "affine_2x3": A.tolist(),
        "table_z_mm": 0.0,
        "image_size": [FRAME_W, FRAME_H],
        "n_markers": len(pairs),
        "residuals_mm": res.tolist(),
        "pixels_px": pixels.tolist(),
        "touched_base_xy_mm": base.tolist(),
        "note": (f"Fitted from {args.color} cube positions (not ArUco). Maps "
                 "pixel -> base XY at CUBE-TOP height, so cube picks absorb the "
                 "parallax that a table-plane fit leaves behind. Slightly off "
                 "for flat objects (cards/coins)."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
