"""
Eye-to-hand calibration for a gantry-mounted camera.

Two ways to compute T_cam2base:

  (A) MARKER ON ARM (capture + solve) — needs ArUco rigidly attached to arm.
  (B) TOUCH CALIBRATION (touch)       — no marker on arm; jog pump tip to
                                        4 corners of a marker on the table.

Usage:
  python gantry_calibrate.py intrinsics  — capture intrinsics with circle grid
  python gantry_calibrate.py touch       — touch-based eye-to-hand (recommended
                                           when you can't attach a marker)
  python gantry_calibrate.py capture     — (method A) move arm to N poses
  python gantry_calibrate.py solve       — (method A) compute T_cam2base
  python gantry_calibrate.py verify      — show marker position in base frame
"""

import sys
import os
import json
import glob
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280

SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0  # gantry USB cam (re-probe after replug if IDs shuffle)
FRAME_W, FRAME_H = 1920, 1080

ARUCO_DICT = cv2.aruco.DICT_6X6_50
MARKER_ID = 3
MARKER_SIZE = 30.0  # mm, side length without white border

# For touch calibration, USE A BIGGER MARKER. Touching the 4 corners of a
# 30mm marker is fiddly and the calibration will be noise-sensitive.
# Print a marker (any id from DICT_6X6_50) at ~80–120mm and set these.
CALIB_MARKER_ID = 3
CALIB_MARKER_SIZE = 93.0  # mm (measured, printer shrank from 100mm)

# Suction pump nozzle length, from TCP (tool flange) to tip, along tool +Z.
PUMP_LENGTH = 70.0  # mm — measured ~7 cm

GRID_SIZE = (5, 7)
CIRCLE_SPACING = 15.0
CIRCLE_FLAGS = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING

DATA_DIR = "gantry_calib"
RESULTS_FILE = "calibration_result.json"


# ── shared helpers ──────────────────────────────────────────────────────

def open_cam():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


def grab(cap, flush=5):
    for _ in range(flush):
        cap.read()
    ret, frame = cap.read()
    return frame if ret else None


def coords_to_T(coords):
    x, y, z, rx, ry, rz = coords
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def detect_marker(frame, mtx, dist, marker_id=MARKER_ID, marker_size=MARKER_SIZE):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(d, p)
    corners, ids, _ = detector.detectMarkers(enhanced)
    if ids is None:
        return None, None, None
    for i, mid in enumerate(ids.ravel()):
        if mid == marker_id:
            half = marker_size / 2.0
            obj_pts = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)
            ret, rvec, tvec = cv2.solvePnP(obj_pts, corners[i], mtx, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ret:
                return rvec, tvec, corners[i]
    return None, None, None


def rigid_align(src, dst):
    """Kabsch/Umeyama: find rigid T such that (T @ [src,1].T)[:3] ≈ dst.
    src, dst: (N,3) arrays of corresponding points."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = dst_c - R @ src_c
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = 10
    p.maxThreshold = 230
    p.thresholdStep = 10
    p.filterByArea = True
    p.minArea = 300       # tighter — was catching keyboard keys / speaker dots
    p.maxArea = 4000      # tighter — your grid dots are ~30-50 px diameter
    p.filterByCircularity = True
    p.minCircularity = 0.6
    p.filterByConvexity = True
    p.minConvexity = 0.8
    p.filterByInertia = True
    p.minInertiaRatio = 0.4
    return cv2.SimpleBlobDetector_create(p)


def make_grid_object_pts():
    cols, rows = GRID_SIZE
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = (2 * c + r % 2) * (CIRCLE_SPACING / 2)
            y = r * CIRCLE_SPACING
            pts.append([x, y, 0])
    return np.array(pts, dtype=np.float32)


# ── intrinsics ──────────────────────────────────────────────────────────

def intrinsics():
    """Hold the circle grid by hand in front of the gantry cam from many angles."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cap = open_cam()
    detector = make_blob_detector()
    objp = make_grid_object_pts()

    obj_points, img_points = [], []
    print("Hold the 5x7 asymmetric circle grid in front of the camera.")
    print("Move it through ~20 poses (different angles, depths, positions).")
    print("Press SPACE to capture, q to finish.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, centers = cv2.findCirclesGrid(gray, GRID_SIZE, None, CIRCLE_FLAGS, detector)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, GRID_SIZE, centers, found)
        cv2.putText(disp, f"captured: {len(obj_points)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("intrinsics", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            obj_points.append(objp)
            img_points.append(centers)
            print(f"  captured #{len(obj_points)}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 10:
        print(f"Only {len(obj_points)} captures — need at least 10."); return

    err, mtx, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, (FRAME_W, FRAME_H), None, None)
    print(f"\nReprojection error: {err:.4f} px")
    print(f"fx={mtx[0,0]:.1f} fy={mtx[1,1]:.1f}  cx={mtx[0,2]:.1f} cy={mtx[1,2]:.1f}")

    intr_path = os.path.join(DATA_DIR, "intrinsics.json")
    with open(intr_path, "w") as f:
        json.dump({
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.tolist(),
            "image_size": [FRAME_W, FRAME_H],
            "reprojection_error_px": err,
        }, f, indent=2)
    print(f"Saved → {intr_path}")


# ── capture poses ───────────────────────────────────────────────────────

# Vary translation AND rotation so the hand-eye solve is well-conditioned.
# These are mycobot280 TCP coords [x,y,z,rx,ry,rz] in mm/deg. Tune to fit
# your gantry's FOV — the marker on the gripper must stay visible.
CAPTURE_POSES = [
    # mostly upright, varied XY heights
    [200, -23, 260, -170,  -4, -42],
    [200,  20, 260, -170,  -4, -42],
    [200, -70, 260, -170,  -4, -42],
    [160, -23, 260, -170,  -4, -42],
    [240, -23, 260, -170,  -4, -42],
    [200, -23, 220, -170,  -4, -42],
    [200, -23, 300, -170,  -4, -42],
    # tilts so the marker plane normal changes (critical for hand-eye)
    [200, -23, 260, -150,  -4, -42],
    [200, -23, 260,  170,  -4, -42],
    [200, -23, 260, -170,  20, -42],
    [200, -23, 260, -170, -20, -42],
    [200, -23, 260, -160,  15,   0],
    [200, -23, 260, -160, -15,   0],
    # combined translation + tilt
    [180, -60, 270, -155,  15, -30],
    [220,  20, 240, -160, -10,  20],
    [180,  20, 280,  170,  10, -60],
    [220, -60, 240, -170, -10,  60],
    [200, -40, 290, -170,  10, -90],
    [200, -40, 230, -170, -10,  60],
    [180, -23, 260, -170,   0,   0],
]


ASSUMED_HFOV_DEG = 60.0  # rough guess for a typical USB webcam


def synthetic_intrinsics(w, h, hfov_deg=ASSUMED_HFOV_DEG):
    """Pinhole camera with no distortion, derived from assumed horizontal FOV.
    Wrong but in the ballpark — use only when proper calibration isn't available."""
    fx = (w / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    fy = fx  # assume square pixels
    cx, cy = w / 2.0, h / 2.0
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    dist = np.zeros((1, 5), dtype=float)
    return mtx, dist


def load_intrinsics():
    intr_path = os.path.join(DATA_DIR, "intrinsics.json")
    if os.path.exists(intr_path):
        with open(intr_path) as f:
            d = json.load(f)
        return np.array(d["camera_matrix"]), np.array(d["dist_coeffs"])
    # No intrinsics file → synthesize from assumed FOV.
    print("⚠  No intrinsics file found — using SYNTHETIC defaults")
    print(f"   assumed HFOV={ASSUMED_HFOV_DEG}°, no distortion")
    print(f"   results will be biased; run `intrinsics` for proper calibration")
    return synthetic_intrinsics(FRAME_W, FRAME_H)


def capture():
    os.makedirs(DATA_DIR, exist_ok=True)
    mtx, dist = load_intrinsics()

    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    mc.power_on()
    time.sleep(0.5)

    cap = open_cam()
    samples = []

    for i, pose in enumerate(CAPTURE_POSES):
        print(f"[{i+1}/{len(CAPTURE_POSES)}] → {pose}")
        try:
            mc.send_coords(pose, 25)
        except Exception as e:
            print(f"  rejected: {e}"); continue
        time.sleep(3.5)

        tcp = mc.get_coords()
        if not tcp or len(tcp) != 6:
            print("  no TCP"); continue

        frame = grab(cap)
        if frame is None:
            print("  no frame"); continue

        rvec, tvec, _ = detect_marker(frame, mtx, dist)
        if rvec is None:
            print("  marker not detected — skipping")
            cv2.imwrite(os.path.join(DATA_DIR, f"miss_{i:02d}.png"), frame)
            continue

        cv2.imwrite(os.path.join(DATA_DIR, f"img_{len(samples):02d}.png"), frame)
        samples.append({
            "index": len(samples),
            "tcp": list(tcp),
            "rvec": rvec.ravel().tolist(),
            "tvec": tvec.ravel().tolist(),
        })
        print(f"  ✓ saved  tvec(cam)={[round(v,1) for v in tvec.ravel()]}")

    cap.release()
    with open(os.path.join(DATA_DIR, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n{len(samples)}/{len(CAPTURE_POSES)} valid samples → {DATA_DIR}/samples.json")


# ── solve ──────────────────────────────────────────────────────────────

def solve():
    mtx, dist = load_intrinsics()
    with open(os.path.join(DATA_DIR, "samples.json")) as f:
        samples = json.load(f)
    if len(samples) < 5:
        print("Need ≥5 samples"); return

    # Eye-to-hand: solve T_cam2base.
    # OpenCV's calibrateHandEye computes T_cam2gripper given T_gripper2base
    # and T_target2cam. For eye-to-hand, we feed it the INVERSE:
    #   T_base2gripper instead of T_gripper2base
    # and the same T_target2cam, and the output is T_cam2base.
    R_b2g, t_b2g, R_t2c, t_t2c = [], [], [], []
    for s in samples:
        T_g2b = coords_to_T(s["tcp"])
        T_b2g = invert_T(T_g2b)
        R_b2g.append(T_b2g[:3, :3])
        t_b2g.append(T_b2g[:3, 3].reshape(3, 1))
        R, _ = cv2.Rodrigues(np.array(s["rvec"]))
        R_t2c.append(R)
        t_t2c.append(np.array(s["tvec"]).reshape(3, 1))

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {}
    print(f"\n── Eye-to-hand solve ({len(samples)} samples) ──")
    for name, m in methods.items():
        R_c2b, t_c2b = cv2.calibrateHandEye(R_b2g, t_b2g, R_t2c, t_t2c, method=m)
        det = float(np.linalg.det(R_c2b))
        print(f"  {name:11s}  t={np.round(t_c2b.ravel(),1)}  det(R)={det:.6f}")
        results[name] = (R_c2b, t_c2b)

    # pick method with det(R) closest to 1
    best = min(results, key=lambda n: abs(np.linalg.det(results[n][0]) - 1.0))
    R_c2b, t_c2b = results[best]
    translations = np.array([results[n][1].ravel() for n in results])
    spread = translations.std(axis=0)
    print(f"\n  Method translation spread: {np.round(spread, 2)} mm")
    print(f"  Chose: {best}")

    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_c2b
    T_cam2base[:3, 3] = t_c2b.ravel()

    # Reprojection sanity check: marker pos in base should be ~constant
    # for fixed marker-on-gripper offset across all samples.
    marker_in_base = []
    for s in samples:
        T_m2c = np.eye(4)
        T_m2c[:3, :3], _ = cv2.Rodrigues(np.array(s["rvec"])), None
        T_m2c[:3, :3], _ = cv2.Rodrigues(np.array(s["rvec"]))
        T_m2c[:3, 3] = np.array(s["tvec"])
        T_m2base = T_cam2base @ T_m2c
        T_g2b = coords_to_T(s["tcp"])
        T_m2g = invert_T(T_g2b) @ T_m2base
        marker_in_base.append(T_m2g[:3, 3])
    arr = np.array(marker_in_base)
    print(f"\n  Marker-on-gripper offset across samples:")
    print(f"    mean = {arr.mean(axis=0).round(2)} mm")
    print(f"    std  = {arr.std(axis=0).round(2)} mm  ← lower is better")

    result = {
        "mode": "eye_to_hand",
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": [FRAME_W, FRAME_H],
        "hand_eye_method": best,
        "T_cam2base": T_cam2base.tolist(),
        "marker_in_gripper_mean_mm": arr.mean(axis=0).tolist(),
        "marker_in_gripper_std_mm": arr.std(axis=0).tolist(),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")


# ── touch calibration (no marker on arm) ───────────────────────────────

CORNER_LABELS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]


def touch():
    """Eye-to-hand calibration by touching the 4 corners of a flat marker
    on the table with the pump tip. No marker needs to be attached to the arm."""
    mtx, dist = load_intrinsics()

    print(f"Using calibration marker: id={CALIB_MARKER_ID}, size={CALIB_MARKER_SIZE}mm")
    print(f"Pump nozzle length: {PUMP_LENGTH}mm  (from TCP flange to suction tip)\n")

    # Open camera now; connect to robot LATER (after marker is locked in).
    # If we connect now, the serial often dies while the user positions the marker.
    cap = open_cam()

    # ── step 1: detect marker on table ──
    print("=== Step 1: detect marker on table ===")
    print("Place the ArUco marker flat on the table inside the camera FOV.")
    print("A live preview will open. Press SPACE when detection is stable, q to abort.\n")

    T_marker2cam = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rvec, tvec, corners = detect_marker(
            frame, mtx, dist, CALIB_MARKER_ID, CALIB_MARKER_SIZE)
        disp = frame.copy()
        if rvec is not None:
            cv2.drawFrameAxes(disp, mtx, dist, rvec, tvec, CALIB_MARKER_SIZE / 2)
            # label the 4 corners so the user knows which to touch first
            for j, label in enumerate(CORNER_LABELS):
                px, py = corners[0][j]
                cv2.circle(disp, (int(px), int(py)), 8, (0, 255, 255), 2)
                cv2.putText(disp, f"{j}:{label}", (int(px) + 12, int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(disp, "SPACE = lock in detection", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "no marker visible", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("touch_calib", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' ') and rvec is not None:
            R_m2c, _ = cv2.Rodrigues(rvec)
            T_marker2cam = np.eye(4)
            T_marker2cam[:3, :3] = R_m2c
            T_marker2cam[:3, 3] = tvec.ravel()
            print(f"Locked. Marker tvec(cam) = {tvec.ravel().round(1)} mm")
            break
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            for _ in range(5): cv2.waitKey(1)
            return

    # On macOS, destroyAllWindows alone leaves a zombie window that freezes
    # the main thread when input() blocks. Pump events to actually close it.
    cv2.destroyAllWindows()
    for _ in range(10): cv2.waitKey(1)
    cap.release()

    # ── step 2: touch each corner ──
    half = CALIB_MARKER_SIZE / 2.0
    corners_marker = np.array([
        [-half,  half, 0],  # 0 TOP-LEFT
        [ half,  half, 0],  # 1 TOP-RIGHT
        [ half, -half, 0],  # 2 BOTTOM-RIGHT
        [-half, -half, 0],  # 3 BOTTOM-LEFT
    ])

    print("\n=== Step 2: touch each corner with the pump tip ===")
    print("Servos will be released so you can move the arm by hand.")
    print("For each corner: position the suction nozzle TIP precisely on the")
    print("printed corner, hold steady, then press ENTER.\n")
    input("Press ENTER to connect to the robot, home it, then release servos...")

    # Connect to robot NOW — fresh serial connection avoids timeout.
    print("  connecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    mc.power_on()
    time.sleep(0.5)

    print("  homing to [0,0,0,0,0,0]...")
    mc.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(4)

    # Now release. Try a few methods — firmware-dependent.
    for fn, args in [("set_free_mode", (1,)), ("release_all_servos", ())]:
        if hasattr(mc, fn):
            try:
                getattr(mc, fn)(*args)
                print(f"  released via {fn}")
            except Exception as e:
                print(f"  {fn} failed: {e}")
    for sid in range(1, 7):
        try:
            mc.release_servo(sid)
        except Exception:
            pass
    time.sleep(1)
    print("  Arm should now be limp. Try moving it by hand.")

    pump_tips_base = []
    for i, label in enumerate(CORNER_LABELS):
        input(f"  [{i}] Move pump tip to {label} corner, then press ENTER...")
        tcps = []
        for _ in range(8):
            t = mc.get_coords()
            if t and len(t) == 6:
                tcps.append(t)
            time.sleep(0.1)
        if len(tcps) < 3:
            print("    Could not read stable TCP — aborting")
            mc.power_on(); cap.release(); return
        tcp = np.mean(tcps, axis=0)

        T_tool2base = coords_to_T(tcp)
        tip_local = np.array([0, 0, PUMP_LENGTH, 1.0])
        tip_base = (T_tool2base @ tip_local)[:3]
        pump_tips_base.append(tip_base)
        print(f"    TCP={tcp[:3].round(1)}  pump_tip(base)={tip_base.round(1)}")

    mc.power_on()
    cap.release()

    pump_tips_base = np.array(pump_tips_base)

    # ── step 3: solve ──
    T_marker2base = rigid_align(corners_marker, pump_tips_base)
    T_cam2base = T_marker2base @ invert_T(T_marker2cam)

    # residual check
    corners_h = np.hstack([corners_marker, np.ones((4, 1))])
    predicted = (T_marker2base @ corners_h.T).T[:, :3]
    residuals = np.linalg.norm(predicted - pump_tips_base, axis=1)
    print(f"\n── Solve ──")
    print(f"Per-corner residual (mm): {residuals.round(2)}")
    print(f"Mean: {residuals.mean():.2f}  Max: {residuals.max():.2f}")
    if residuals.max() > 5.0:
        print("  ⚠  Max residual > 5mm — touches were probably imprecise. Redo.")
    elif residuals.max() > 2.0:
        print("  ⚠  Max residual > 2mm — usable but not great.")
    else:
        print("  ✓ tight fit")

    result = {
        "mode": "eye_to_hand_touch",
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": [FRAME_W, FRAME_H],
        "T_cam2base": T_cam2base.tolist(),
        "touch_residuals_mm": residuals.tolist(),
        "calib_marker": {"id": CALIB_MARKER_ID, "size_mm": CALIB_MARKER_SIZE},
        "pump_length_mm": PUMP_LENGTH,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")


# ── verify ─────────────────────────────────────────────────────────────

def verify():
    """Live: show detected marker position in base frame. Move the marker
    around, sanity-check the numbers against where you think it is."""
    with open(RESULTS_FILE) as f:
        cal = json.load(f)
    mtx = np.array(cal["camera_matrix"])
    dist = np.array(cal["dist_coeffs"])
    T_cam2base = np.array(cal["T_cam2base"])

    cap = open_cam()
    print("Live view. Move marker; (q) to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        rvec, tvec, _ = detect_marker(frame, mtx, dist)
        if rvec is not None:
            T_m2c = np.eye(4)
            T_m2c[:3, :3], _ = cv2.Rodrigues(rvec)
            T_m2c[:3, 3] = tvec.ravel()
            pos = (T_cam2base @ T_m2c)[:3, 3]
            txt = f"base: x={pos[0]:6.1f} y={pos[1]:6.1f} z={pos[2]:6.1f}"
            cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        cv2.imshow("verify", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cmds = {"intrinsics": intrinsics, "touch": touch, "capture": capture,
            "solve": solve, "verify": verify}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__); sys.exit(1)
    cmds[sys.argv[1]]()
