"""
Multi-marker 2D affine calibration.

Scatter ArUco markers (any IDs, DICT_6X6_50) across your workspace,
detect them all in one frame, then touch each marker's CENTER with the
pump tip. Fits a 2D affine from N (pixel_center → base_xy) pairs.

Why this beats the single-marker version:
- Spatial coverage: affine is validated across the actual pick area,
  not just a 10x10cm patch → no extrapolation error at pick time.
- More constraints per calibration (N pairs vs 4 corners of one marker).
- Center-touching is much easier than corner-touching for small markers.

Workflow:
  1. Scatter 5-10 markers across the workspace where you'll pick objects
  2. Run this script
  3. SPACE = lock the currently visible markers and their IDs
  4. Servos release; touch each marker's CENTER with the pump tip in ID order
  5. Script fits the affine and writes calibration_affine2d.json
"""
import json
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280

SERIAL_PORT = "/dev/tty.usbserial-0202EDB8"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
PUMP_LENGTH = 70.0

ARUCO_DICT = cv2.aruco.DICT_6X6_50
OUT_PATH = "/Users/v/Downloads/69conference/calibration_affine2d.json"

MIN_MARKERS = 4          # need at least this many for a reliable affine fit


def detect_all_markers(frame):
    """Return dict of {marker_id: center_pixel, corners}. Empty if none."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(d, p)
    corners, ids, _ = det.detectMarkers(frame)
    out = {}
    if ids is None:
        return out
    for i, mid in enumerate(ids.flatten().tolist()):
        c = corners[i][0]  # (4, 2)
        center = c.mean(axis=0)
        out[int(mid)] = {"center": center, "corners": c}
    return out


def coords_to_T(coords):
    """[x,y,z,rx,ry,rz] (mm/deg) → 4x4 transform, xyz extrinsic convention."""
    x, y, z, rx, ry, rz = coords
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def main():
    print(f"Opening camera id={CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8): cap.read()

    print(f"Scatter markers across your workspace. Aim for {MIN_MARKERS}+ markers.")
    print("SPACE = lock the visible markers   |   Q = quit\n")

    cv2.namedWindow("affine2d_multi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("affine2d_multi", 1280, 720)

    locked = None  # dict of {id: center_pixel}
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        detected = detect_all_markers(frame)
        disp = frame.copy()
        for mid, m in detected.items():
            c = m["corners"]
            for i in range(4):
                cv2.line(disp,
                         tuple(int(x) for x in c[i]),
                         tuple(int(x) for x in c[(i + 1) % 4]),
                         (0, 255, 0), 2)
            cx, cy = int(m["center"][0]), int(m["center"][1])
            cv2.circle(disp, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(disp, f"id {mid}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        color = (0, 200, 0) if len(detected) >= MIN_MARKERS else (0, 165, 255)
        cv2.putText(disp,
                    f"detected: {len(detected)} markers (need >= {MIN_MARKERS})    "
                    f"IDs: {sorted(detected.keys())}    SPACE=lock  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imshow("affine2d_multi", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return
        if k == 32:
            if len(detected) < MIN_MARKERS:
                print(f"  need ≥ {MIN_MARKERS} markers to lock; only {len(detected)} visible")
                continue
            locked = {mid: m["center"].copy() for mid, m in detected.items()}
            locked_frame = frame.copy()
            locked_detected = detected  # keep for saving snapshots per touch
            print(f"\nLocked {len(locked)} markers:")
            for mid in sorted(locked.keys()):
                u, v = locked[mid]
                print(f"  id {mid:2d} → pixel ({u:.1f}, {v:.1f})")
            break

    cv2.destroyAllWindows()
    for _ in range(10): cv2.waitKey(1)
    cap.release()

    # Save a labelled reference snapshot so the user can see marker positions
    # while running through the touches below (input() blocks the cv2 window).
    ref = locked_frame.copy()
    for mid, m in locked_detected.items():
        c = m["corners"]
        for i in range(4):
            cv2.line(ref, tuple(int(x) for x in c[i]),
                     tuple(int(x) for x in c[(i + 1) % 4]), (0, 255, 0), 3)
        cx, cy = int(m["center"][0]), int(m["center"][1])
        cv2.circle(ref, (cx, cy), 12, (0, 0, 255), -1)
        cv2.putText(ref, f"id {mid}", (cx + 15, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    ref_path = "/Users/v/Downloads/69conference/affine_markers.png"
    cv2.imwrite(ref_path, ref)
    print(f"\n✓ Saved labelled snapshot: {ref_path}")
    print("  Open it in Preview so you can see which marker is which ID.")
    try:
        import subprocess
        subprocess.Popen(["open", ref_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # ── Step 2: touch each marker center ──
    print("\n=== Step 2: touch each marker CENTER with the pump tip ===")
    print("Order will follow ID ascending.\n")
    input("Press ENTER to connect to the robot, home it, then release servos...")

    print("  connecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    try:
        mc.power_on(); time.sleep(1.0)
        print("    power_on OK")
    except Exception as e:
        print(f"    power_on failed: {e}")

    # Undo any leftover free-mode / servo-release from a previous crashed run.
    # Without this, send_angles is silently ignored.
    try:
        mc.set_free_mode(0)   # 0 = free mode OFF → servos hold position
        print("    set_free_mode(0) — servos engaged")
    except Exception as e:
        print(f"    set_free_mode(0) failed: {e}")
    try:
        if hasattr(mc, "focus_all_servos"):
            mc.focus_all_servos()
            print("    focus_all_servos — servos engaged")
    except Exception as e:
        print(f"    focus_all_servos failed: {e}")
    time.sleep(0.5)

    before = mc.get_angles()
    print(f"    angles before home: {before!r}")
    print("  homing to [0,0,0,0,0,0]...")
    mc.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(5)
    after = mc.get_angles()
    print(f"    angles after home:  {after!r}")
    released = False
    for fn, args in [("set_free_mode", (1,)),
                     ("release_all_servos", ()),
                     ("focus_all_servos", ())]:  # focus_all_servos(0)-style may release
        if hasattr(mc, fn):
            try:
                r = getattr(mc, fn)(*args)
                print(f"    {fn}({args}) → {r}")
                released = True
            except Exception as e:
                print(f"    {fn} failed: {e}")
    for sid in range(1, 7):
        try:
            r = mc.release_servo(sid)
            print(f"    release_servo({sid}) → {r}")
            released = True
        except Exception as e:
            print(f"    release_servo({sid}) failed: {e}")
    time.sleep(1)
    if released:
        print("\n  Arm SHOULD now be limp — try pushing joint 1 side-to-side.")
        print("  If it's still stiff, some models need mc.power_off() instead.\n")
    else:
        print("\n  ⚠ No release method succeeded. Try running mc.power_off() manually.\n")
    input("  Press ENTER once the arm is limp (or Ctrl-C to abort): ")

    ids_sorted = sorted(locked.keys())
    pairs = []  # list of (pixel_xy, base_xy)

    for mid in ids_sorted:
        u, v = locked[mid]
        prompt = f"  id {mid:2d} (pixel {u:.0f},{v:.0f}) — touch its CENTER, then ENTER"
        cmd = input(f"{prompt}  (or type 's' to skip): ").strip().lower()
        if cmd == "s":
            print(f"    skipped id {mid}")
            continue
        tcps = []
        for _ in range(8):
            t = mc.get_coords()
            if isinstance(t, (list, tuple)) and len(t) == 6:
                tcps.append(t)
            time.sleep(0.1)
        if len(tcps) < 3:
            print(f"    ✗ couldn't read stable TCP for id {mid} — skipping")
            continue
        tcp = np.mean(tcps, axis=0)
        T_flange2base = coords_to_T(tcp)
        tip_local = np.array([0, 0, PUMP_LENGTH, 1.0])
        tip_base = (T_flange2base @ tip_local)[:3]
        pairs.append(((u, v), tip_base[:2]))
        print(f"    ✓ pump_tip_xy(base)={tip_base[:2].round(1)}")

    if len(pairs) < MIN_MARKERS:
        print(f"\n✗ only {len(pairs)} touches, need ≥ {MIN_MARKERS}. Aborting.")
        return

    # ── Step 3: fit affine ──
    print("\n── Solve ──")
    pixels = np.array([p[0] for p in pairs], dtype=np.float32)
    base = np.array([p[1] for p in pairs], dtype=np.float32)
    # Plain least-squares fit — every point contributes equally, no outlier rejection.
    # (LMEDS with only 4 points was rejecting one and reporting 0-residual for the
    # remaining 3, which is misleading.)
    n = len(pixels)
    P = np.hstack([pixels, np.ones((n, 1), dtype=np.float32)])  # (n, 3)
    # Solve A^T (3x2) such that P @ A^T ≈ base. Then A = (A^T).T is (2, 3).
    A_T, _, _, _ = np.linalg.lstsq(P, base, rcond=None)
    A = A_T.T.astype(np.float32)
    if A is None:
        print("  ✗ affine fit failed")
        return

    pixels_h = np.hstack([pixels, np.ones((len(pixels), 1), dtype=np.float32)])
    predicted = (A @ pixels_h.T).T
    residuals = np.linalg.norm(predicted - base, axis=1)
    print(f"  Fit from {len(pairs)} markers")
    print(f"  A =")
    print(f"    [{A[0,0]:+.6f}  {A[0,1]:+.6f}  {A[0,2]:+.2f}]")
    print(f"    [{A[1,0]:+.6f}  {A[1,1]:+.6f}  {A[1,2]:+.2f}]")
    print(f"  Per-marker residual (mm):")
    for i, (mid, r) in enumerate(zip([p[0] for p in zip(ids_sorted, pairs)], residuals)):
        print(f"    id {mid[0] if isinstance(mid, tuple) else mid:2d}  res={r:.3f}")
    print(f"  Mean: {residuals.mean():.3f}   Max: {residuals.max():.3f}")

    out = {
        "mode": "affine2d_pixel_to_base_xy_multi",
        "affine_2x3": A.tolist(),
        "table_z_mm": 0.0,
        "image_size": [FRAME_W, FRAME_H],
        "n_markers": len(pairs),
        "residuals_mm": residuals.tolist(),
        "pixels_px": pixels.tolist(),
        "touched_base_xy_mm": base.tolist(),
        "note": ("2D affine from multiple scattered markers. Better coverage "
                 "than single-marker version. Table-plane only (z ≈ 0)."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
