"""
Pattern-based eye-to-hand calibration.

Mount the chessboard rigidly on the end-effector (any orientation — the offset
is solved for). Camera must be FIXED to the gantry (this is eye-to-hand).

Workflow:
  - Run the script. Servos release so you can drag the arm.
  - For each pose: position the arm so the chessboard is fully in view,
    hold steady, press SPACE.
  - Repeat for >= 12 poses. Aim for diverse rotations on ALL 3 axes,
    not just translations.
  - Press ENTER to solve and write a new calibration_result.json.

Keys (in the preview window):
  SPACE = capture this pose
  U     = undo last pose
  ENTER = solve and save
  Q     = quit without saving

If chessboard squares didn't print at exactly 20 mm, update SQUARE_MM below.
"""
import json
import os
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280
from ikpy.chain import Chain

# ── config ──────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.usbserial-54780106801"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080

URDF_PATH = "/Users/v/Downloads/69conference/mycobot_280_m5.urdf"
INTRINSICS_PATH = "/Users/v/Downloads/69conference/gantry_calib/intrinsics.json"
CALIB_PATH = "/Users/v/Downloads/69conference/calibration_result.json"

# Chessboard — UPDATE SQUARE_MM if your printed pattern measured differently.
BOARD_COLS, BOARD_ROWS = 9, 6   # inner corners
SQUARE_MM = 14.0
MIN_POSES = 10


# ── ikpy chain (matches qwen_command.py exactly) ───────────────────
chain = Chain.from_urdf_file(
    URDF_PATH, base_elements=['g_base'], last_link_vector=[0, 0, 0],
    active_links_mask=[False, False, True, True, True, True, True, True, False],
)


def fk_g2b(angles_deg):
    """Forward kinematics: 6 joint angles in degrees → 4x4 base-to-flange (mm)."""
    pose = [0.0] * len(chain.links)
    for i, deg in enumerate(angles_deg):
        pose[i + 2] = float(np.radians(deg))
    T = chain.forward_kinematics(pose)
    T_mm = T.copy()
    T_mm[:3, 3] *= 1000.0
    return T_mm


def load_intrinsics():
    if os.path.exists(INTRINSICS_PATH):
        with open(INTRINSICS_PATH) as f: d = json.load(f)
    elif os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f: d = json.load(f)
    else:
        raise RuntimeError("No intrinsics found in either file")
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1)
    return K, dist


def detect_board(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags=flags)
    if not ok:
        return False, None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return True, corners


def pattern_pose(corners, K, dist):
    """Solve PnP: pattern → camera. Returns (R, t) with t in mm."""
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_MM
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def solve_handeye(R_g2b_all, t_g2b_all, R_p2c_all, t_p2c_all):
    """
    Eye-to-hand trick: pass base→gripper (inverse of gripper→base) and the
    output is cam→base instead of cam→gripper.
    """
    R_b2g_all = [Rg.T for Rg in R_g2b_all]
    t_b2g_all = [-Rg.T @ tg for Rg, tg in zip(R_g2b_all, t_g2b_all)]

    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = []
    for name, m in methods.items():
        R_c2b, t_c2b = cv2.calibrateHandEye(
            R_b2g_all, t_b2g_all, R_p2c_all, t_p2c_all, method=m)
        T = np.eye(4)
        T[:3, :3] = R_c2b
        T[:3, 3] = t_c2b.reshape(3)

        # Consistency check: the pattern is rigidly attached to the flange,
        # so pattern-origin expressed in flange-frame should be constant across poses.
        in_grip = []
        for Rg, tg, Rp, tp in zip(R_g2b_all, t_g2b_all, R_p2c_all, t_p2c_all):
            p_base = T[:3, :3] @ tp + T[:3, 3]
            p_grip = Rg.T @ (p_base - tg)
            in_grip.append(p_grip)
        in_grip = np.array(in_grip)
        residual = float(np.linalg.norm(in_grip - in_grip.mean(axis=0), axis=1).mean())

        cam = T[:3, 3]
        print(f"  {name:10s} cam=({cam[0]:7.1f}, {cam[1]:7.1f}, {cam[2]:7.1f})  "
              f"residual={residual:.2f} mm")
        results.append((name, T, residual))

    results.sort(key=lambda r: r[2])
    return results[0]


def save_calib(T_cam2base, K, dist, method, residual, n_poses):
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f: out = json.load(f)
        backup = CALIB_PATH + ".touch.bak"
        if not os.path.exists(backup):
            with open(backup, "w") as f: json.dump(out, f, indent=2)
            print(f"Backed up old calibration → {backup}")
    else:
        out = {}

    out["mode"] = "eye_to_hand_pattern"
    out["camera_matrix"] = K.tolist()
    out["dist_coeffs"] = [dist.tolist()]
    out["image_size"] = [FRAME_W, FRAME_H]
    out["T_cam2base"] = T_cam2base.tolist()
    out["handeye_method"] = method
    out["handeye_residual_mm"] = residual
    out["handeye_pose_count"] = n_poses
    out["chessboard"] = {
        "inner_cols": BOARD_COLS, "inner_rows": BOARD_ROWS, "square_mm": SQUARE_MM,
    }
    out.pop("touch_residuals_mm", None)
    out.pop("calib_marker", None)

    with open(CALIB_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {CALIB_PATH}")


def main():
    print(f"Loading intrinsics from {INTRINSICS_PATH if os.path.exists(INTRINSICS_PATH) else CALIB_PATH}")
    K, dist = load_intrinsics()
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    print(f"Opening camera id={CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8): cap.read()

    print(f"Connecting robot at {SERIAL_PORT}...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(1.0)
    mc.release_all_servos()
    print("Servos released. Drag the arm into pose, hold steady, press SPACE.")
    print()
    print("Pose strategy: 12-20 poses with tilt on ALL three rotation axes.")
    print("Don't just slide the arm flat — twist the wrist, rotate around the pattern normal.")
    print()

    R_g2b_all, t_g2b_all = [], []
    R_p2c_all, t_p2c_all = [], []
    captures_meta = []
    cv2.namedWindow("handeye", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("handeye", 1280, 720)

    last_corners = None
    last_detect_ok = False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_board(gray)
        last_corners = corners
        last_detect_ok = ok

        disp = frame.copy()
        if ok:
            cv2.drawChessboardCorners(disp, (BOARD_COLS, BOARD_ROWS), corners, ok)
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.putText(disp,
                    f"poses: {len(R_g2b_all)}    "
                    f"board: {'DETECTED' if ok else 'no'}    "
                    f"SPACE=cap  U=undo  ENTER=solve  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("handeye", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Quitting without solve.")
            cap.release(); cv2.destroyAllWindows()
            return
        if k == ord('u'):
            if R_g2b_all:
                R_g2b_all.pop(); t_g2b_all.pop()
                R_p2c_all.pop(); t_p2c_all.pop()
                meta = captures_meta.pop()
                print(f"  ← undo pose {meta['idx']}  (now {len(R_g2b_all)} total)")
            continue
        if k == 32:  # SPACE
            if not last_detect_ok:
                print("  ✗ no chessboard in view")
                continue
            # Bracket joint reads around a fresh frame capture so we catch wobble.
            # Camera buffers ~3 frames on macOS USB — discard stale ones first.
            for _ in range(3):
                cap.grab()
            j1 = mc.get_angles()
            ret2, fresh = cap.read()
            j2 = mc.get_angles()
            if not ret2:
                print("  ✗ camera read failed")
                continue
            if not j1 or not j2 or len(j1) != 6 or len(j2) != 6 or all(a == 0 for a in j1):
                print(f"  ✗ bad joint read: j1={j1} j2={j2}")
                continue
            drift = max(abs(a - b) for a, b in zip(j1, j2))
            if drift > 1.0:
                print(f"  ✗ arm wobbled {drift:.2f}° during capture — hold steadier")
                continue
            joints = [(a + b) / 2.0 for a, b in zip(j1, j2)]
            # Detect chessboard on the synced frame, not the preview frame.
            gray2 = cv2.cvtColor(fresh, cv2.COLOR_BGR2GRAY)
            ok2, corners2 = detect_board(gray2)
            if not ok2:
                print("  ✗ chessboard not in synced frame (preview was stale)")
                continue
            try:
                T_g2b = fk_g2b(joints)
            except Exception as e:
                print(f"  ✗ FK failed: {e}")
                continue
            R_p, t_p = pattern_pose(corners2, K, dist)
            if R_p is None:
                print("  ✗ PnP failed")
                continue
            R_g2b_all.append(T_g2b[:3, :3])
            t_g2b_all.append(T_g2b[:3, 3].copy())
            R_p2c_all.append(R_p)
            t_p2c_all.append(t_p)
            tip = T_g2b[:3, 3]
            captures_meta.append({"idx": len(R_g2b_all), "joints": list(joints)})
            print(f"  ✓ pose {len(R_g2b_all):2d}  "
                  f"joints=[{','.join(f'{a:+5.1f}' for a in joints)}]  "
                  f"tip=({tip[0]:6.1f},{tip[1]:6.1f},{tip[2]:6.1f})  "
                  f"pattern_cam=({t_p[0]:6.1f},{t_p[1]:6.1f},{t_p[2]:6.1f})")
            continue
        if k in (13, 10):  # ENTER
            if len(R_g2b_all) < MIN_POSES:
                print(f"  need ≥ {MIN_POSES} poses, have {len(R_g2b_all)}")
                continue
            print(f"\nSolving with {len(R_g2b_all)} poses across 5 methods:")
            name, T_cam2base, residual = solve_handeye(
                R_g2b_all, t_g2b_all, R_p2c_all, t_p2c_all)
            print(f"\nBest: {name}  residual={residual:.2f} mm")

            cam = T_cam2base[:3, 3]
            print(f"  camera position in base frame: ({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f}) mm")

            if os.path.exists(CALIB_PATH):
                with open(CALIB_PATH) as f: old = json.load(f)
                if "T_cam2base" in old:
                    old_cam = np.array(old["T_cam2base"])[:3, 3]
                    delta = np.linalg.norm(cam - old_cam)
                    print(f"  delta vs previous calibration: {delta:.1f} mm")

            save_calib(T_cam2base, K, dist, name, residual, len(R_g2b_all))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
