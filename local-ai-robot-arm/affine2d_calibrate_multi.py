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
import os
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280

SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
PUMP_LENGTH = 70.0

ARUCO_DICT = cv2.aruco.DICT_6X6_50
OUT_PATH = "/Users/v/local-ai-robot-arm/calibration_affine2d.json"

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


def _reconnect(old_mc=None):
    """Re-open the serial connection after a USB drop. Returns a fresh mc.
    Only re-opens serial — does NOT power on or re-focus servos, so the arm
    stays limp for touch calibration."""
    try:
        if old_mc is not None and hasattr(old_mc, "_serial_port"):
            old_mc._serial_port.close()
    except Exception:
        pass
    # Wait for the port to re-enumerate after the USB drop.
    for _ in range(40):
        if os.path.exists(SERIAL_PORT):
            break
        time.sleep(0.5)
    time.sleep(1.0)
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(1.0)
    return mc


def _safe_get_coords(mc):
    """get_coords() that survives a transient USB serial drop (the arm gets
    dragged during touch cal and tugs the cable). Returns (coords_or_None, mc);
    mc may be a reconnected handle. Keep holding the arm at the marker while it
    reconnects."""
    try:
        return mc.get_coords(), mc
    except Exception as e:
        print(f"\n  ⚠ serial dropped ({e}).")
        input("    KEEP HOLDING the arm at the marker. Check the USB cable, "
              "then press ENTER to reconnect...")
        try:
            mc = _reconnect(mc)
            print("    ✓ reconnected.")
            return mc.get_coords(), mc
        except Exception as e2:
            print(f"    ✗ reconnect failed ({e2})")
            return None, mc


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
    # keep `cap` OPEN — the touch step below shows a live view that highlights
    # each marker, so you never have to leave the window.

    # Save a labelled reference snapshot as a record (the live touch view below
    # highlights each marker, so there's no need to open this).
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
    ref_path = "/Users/v/local-ai-robot-arm/affine_markers.png"
    cv2.imwrite(ref_path, ref)
    print(f"\n✓ Saved labelled snapshot (record only): {ref_path}")

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
    pairs = []          # list of (pixel_xy, base_xy)
    recorded_ids = []   # marker id for each entry in `pairs`, in order

    print("\n=== Step 2 (live): touch the highlighted marker, then press SPACE ===")
    print("  In the window:  SPACE = record   S = skip   Q = finish/quit\n")

    win = "affine2d_touch"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    def draw_touch_view(frame, target_mid, flash=None):
        disp = frame.copy()
        # all other locked markers: green if recorded, gray if pending
        for m2 in ids_sorted:
            if m2 == target_mid:
                continue
            cu, cvp = int(locked[m2][0]), int(locked[m2][1])
            col = (0, 180, 0) if m2 in recorded_ids else (150, 150, 150)
            cv2.circle(disp, (cu, cvp), 9, col, 2)
            cv2.putText(disp, f"{m2}", (cu + 11, cvp - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        # current target: big pulsing red target + crosshair + label
        cx, cy = int(locked[target_mid][0]), int(locked[target_mid][1])
        r = 24 + int(8 * abs(np.sin(time.time() * 4.0)))
        cv2.circle(disp, (cx, cy), r, (0, 0, 255), 3)
        cv2.drawMarker(disp, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 44, 2)
        cv2.putText(disp, f"TOUCH ID {target_mid}", (cx + 20, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # header
        cv2.putText(disp,
                    f"[{len(recorded_ids)}/{len(ids_sorted)} recorded]   "
                    f"now: ID {target_mid}    SPACE=record  S=skip  Q=finish",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if flash:
            cv2.putText(disp, flash, (20, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return disp

    idx = 0
    flash = None
    quit_early = False
    while idx < len(ids_sorted):
        mid = ids_sorted[idx]
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        cv2.imshow(win, draw_touch_view(frame, mid, flash))
        k = cv2.waitKey(30) & 0xFF

        if k == ord('q'):
            quit_early = True
            break
        if k == ord('s'):
            print(f"  skipped id {mid}")
            flash = None
            idx += 1
            continue
        if k == 32:  # SPACE → record this marker
            tcps = []
            for _ in range(8):
                t, mc = _safe_get_coords(mc)
                if isinstance(t, (list, tuple)) and len(t) == 6:
                    tcps.append(t)
                time.sleep(0.1)
            if len(tcps) < 3:
                print(f"  ✗ couldn't read stable TCP for id {mid} — try again")
                flash = f"couldn't read arm pose for ID {mid}, try again"
                continue
            tcp = np.mean(tcps, axis=0)
            T_flange2base = coords_to_T(tcp)
            tip_base = (T_flange2base @ np.array([0, 0, PUMP_LENGTH, 1.0]))[:3]
            pairs.append((locked[mid], tip_base[:2]))
            recorded_ids.append(mid)
            print(f"  ✓ id {mid}: pump_tip_xy(base)={tip_base[:2].round(1)}")
            flash = f"recorded ID {mid}"
            idx += 1

    cv2.destroyAllWindows()
    for _ in range(10): cv2.waitKey(1)
    cap.release()

    if quit_early:
        print(f"\n  finished early — {len(pairs)} markers recorded.")

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
    for mid, r in zip(recorded_ids, residuals):
        print(f"    id {mid:2d}  res={r:.3f}")
    print(f"  Mean: {residuals.mean():.3f}   Max: {residuals.max():.3f}")

    out = {
        "mode": "affine2d_pixel_to_base_xy_multi",
        "affine_2x3": A.tolist(),
        "table_z_mm": 0.0,
        "image_size": [FRAME_W, FRAME_H],
        "n_markers": len(pairs),
        "marker_ids": recorded_ids,
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
