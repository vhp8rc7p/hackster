"""
2D affine calibration — pixel (u,v) → base-frame (x,y) at table level.

This is a deliberate SIMPLIFICATION of the full 3D hand-eye calibration:
- No PnP, no marker-size assumption, no FK
- Just 4 touched corners + 4 detected pixel positions → 2x3 affine matrix

Only valid for objects AT the calibrated Z plane (the table surface).
For thick objects (cubes) or held objects (hand), you'd need a separate
affine per Z, or fall back to the 3D calibration.

Workflow:
  1. Place ArUco marker id 3 on the desk (same one used for touch cal)
  2. Run this script. It opens the camera, detects the marker.
  3. Press SPACE to lock the marker pixel positions.
  4. Servos release. Touch each corner with the pump tip, press ENTER.
  5. Script fits the affine and writes calibration_affine2d.json.
"""
import json
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280

# ── config ──────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.usbserial-0202EDB8"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
PUMP_LENGTH = 70.0

MARKER_ID = 3
ARUCO_DICT = cv2.aruco.DICT_6X6_50
OUT_PATH = "/Users/v/Downloads/69conference/calibration_affine2d.json"

CORNER_LABELS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]


def detect_marker_corners(frame, marker_id):
    """Return 4 pixel coordinates of the marker corners, in the OpenCV order:
    [TL, TR, BR, BL]. Returns None if marker not seen."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(d, p)
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None:
        return None
    for i, mid in enumerate(ids.flatten().tolist()):
        if int(mid) == marker_id:
            # corners[i] is shape (1, 4, 2); rows are TL, TR, BR, BL
            return corners[i][0]
    return None


def coords_to_T(coords):
    """[x,y,z,rx,ry,rz] (mm/deg) → 4x4 transform using xyz extrinsic (confirmed convention)."""
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

    print(f"Place ArUco id={MARKER_ID} flat on the desk.")
    print("SPACE = lock marker pixel positions  |  Q = quit\n")

    cv2.namedWindow("affine2d", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("affine2d", 1280, 720)

    locked_corners_px = None
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        pixel_corners = detect_marker_corners(frame, MARKER_ID)
        disp = frame.copy()
        if pixel_corners is not None:
            for i in range(4):
                p1 = tuple(int(x) for x in pixel_corners[i])
                p2 = tuple(int(x) for x in pixel_corners[(i + 1) % 4])
                cv2.line(disp, p1, p2, (0, 255, 0), 2)
                cv2.putText(disp, f"{i}:{CORNER_LABELS[i]}", p1,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        status = (f"marker {'OK' if pixel_corners is not None else 'lost'}    "
                  f"SPACE=lock  Q=quit")
        cv2.putText(disp, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if pixel_corners is not None else (0, 0, 255), 2)
        cv2.imshow("affine2d", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cap.release(); cv2.destroyAllWindows(); return
        if k == 32 and pixel_corners is not None:
            locked_corners_px = pixel_corners.copy()
            print(f"Locked corners (px):")
            for i, (u, v) in enumerate(locked_corners_px):
                print(f"  [{i}] {CORNER_LABELS[i]:13s} ({u:.1f}, {v:.1f})")
            break

    cv2.destroyAllWindows()
    for _ in range(10): cv2.waitKey(1)
    cap.release()

    # ── Step 2: touch each corner ──
    print("\n=== Step 2: touch each corner with the pump tip ===")
    print("Pump should be ATTACHED; we use it to physically touch corners.\n")
    input("Press ENTER to connect to the robot, home it, then release servos...")

    print("  connecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2); mc.power_on(); time.sleep(0.5)
    print("  homing to [0,0,0,0,0,0]...")
    mc.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(4)
    for fn, args in [("set_free_mode", (1,)), ("release_all_servos", ())]:
        if hasattr(mc, fn):
            try: getattr(mc, fn)(*args)
            except Exception: pass
    for sid in range(1, 7):
        try: mc.release_servo(sid)
        except Exception: pass
    time.sleep(1)
    print("  Arm should now be limp. Try moving it by hand.\n")

    touched_base = []
    for i, label in enumerate(CORNER_LABELS):
        input(f"  [{i}] Move pump tip to {label} corner, then press ENTER...")
        tcps = []
        for _ in range(8):
            t = mc.get_coords()
            if isinstance(t, (list, tuple)) and len(t) == 6:
                tcps.append(t)
            time.sleep(0.1)
        if len(tcps) < 3:
            print("    ✗ couldn't read stable TCP — aborting")
            mc.power_on(); return
        tcp = np.mean(tcps, axis=0)
        T_flange2base = coords_to_T(tcp)
        tip_local = np.array([0, 0, PUMP_LENGTH, 1.0])
        tip_base = (T_flange2base @ tip_local)[:3]
        touched_base.append(tip_base[:2])  # XY only
        print(f"    TCP={np.array(tcp[:3]).round(1)}  pump_tip_xy(base)={tip_base[:2].round(1)}")

    touched_base = np.array(touched_base, dtype=np.float32)
    pixels = locked_corners_px.astype(np.float32)

    # ── Step 3: solve affine ──
    print("\n── Solve ──")
    # cv2.estimateAffine2D fits a 2x3 affine [a,b,c; d,e,f] such that
    # [x_base, y_base]^T = A @ [u, v, 1]^T
    A, inliers = cv2.estimateAffine2D(pixels, touched_base, method=cv2.LMEDS)
    if A is None:
        print("  ✗ affine fit failed")
        return

    # Residuals: how well does the affine fit the 4 corners?
    pixels_h = np.hstack([pixels, np.ones((4, 1), dtype=np.float32)])
    predicted = (A @ pixels_h.T).T
    residuals_mm = np.linalg.norm(predicted - touched_base, axis=1)
    print(f"  Affine matrix A (2x3):")
    print(f"    [{A[0,0]:+8.5f}  {A[0,1]:+8.5f}  {A[0,2]:+8.2f}]")
    print(f"    [{A[1,0]:+8.5f}  {A[1,1]:+8.5f}  {A[1,2]:+8.2f}]")
    print(f"  Per-corner residual (mm): {residuals_mm.round(3)}")
    print(f"  Mean: {residuals_mm.mean():.3f}  Max: {residuals_mm.max():.3f}")

    out = {
        "mode": "affine2d_pixel_to_base_xy",
        "affine_2x3": A.tolist(),
        "table_z_mm": 0.0,
        "image_size": [FRAME_W, FRAME_H],
        "marker_id": MARKER_ID,
        "pump_length_mm": PUMP_LENGTH,
        "residuals_mm": residuals_mm.tolist(),
        "pixels_px": pixels.tolist(),
        "touched_base_xy_mm": touched_base.tolist(),
        "note": ("2D affine, table-plane only. Apply with: "
                 "[x,y] = A @ [u,v,1]. Do NOT use for objects at non-zero Z.")
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
