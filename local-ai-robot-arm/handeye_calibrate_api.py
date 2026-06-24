"""
Pattern-based eye-to-hand calibration — using mycobot's BUILT-IN API only.
No ikpy, no URDF. Uses `mc.get_coords()` for gripper pose.

This is a parallel to handeye_calibrate.py for comparison. Output goes to
calibration_result_api.json (separate file) so you can compare the two
T_cam2base side by side.

Workflow is identical: drag the arm, press SPACE per pose, ENTER to solve.

Because mycobot's pymycobot library doesn't document the Euler-angle
convention used by `get_coords()`, this script tries all 12 standard
conventions and reports which one gives the lowest residual. The winning
convention is mycobot's actual convention.
"""
import json
import os
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pymycobot.mycobot280 import MyCobot280

# ── config ──────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.usbserial-54780106801"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080

INTRINSICS_PATH = "/Users/v/Downloads/69conference/gantry_calib/intrinsics.json"
EXISTING_CALIB_PATH = "/Users/v/Downloads/69conference/calibration_result.json"
OUT_PATH = "/Users/v/Downloads/69conference/calibration_result_api.json"

# Chessboard — must match handeye_calibrate.py
BOARD_COLS, BOARD_ROWS = 9, 6
SQUARE_MM = 14.0
MIN_POSES = 10

# Euler conventions to try. Lowercase = extrinsic, uppercase = intrinsic.
EULER_CONVENTIONS = [
    "xyz", "xzy", "yxz", "yzx", "zxy", "zyx",
    "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
]


def load_intrinsics():
    if os.path.exists(INTRINSICS_PATH):
        with open(INTRINSICS_PATH) as f: d = json.load(f)
    elif os.path.exists(EXISTING_CALIB_PATH):
        with open(EXISTING_CALIB_PATH) as f: d = json.load(f)
    else:
        raise RuntimeError("No intrinsics found")
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
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2) * SQUARE_MM
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    Rmat, _ = cv2.Rodrigues(rvec)
    return Rmat, tvec.reshape(3)


def read_coords_stable(mc, tries=4):
    """get_coords() is flaky; retry until we get a non-zero reading."""
    for _ in range(tries):
        c = mc.get_coords()
        if c and len(c) == 6 and not all(v == 0 for v in c):
            return c
        time.sleep(0.1)
    return None


def coords_to_T(coords, convention):
    """[x,y,z,rx,ry,rz] (mm, degrees) → 4x4 transform using the given Euler convention."""
    xyz = np.array(coords[:3], dtype=np.float64)
    rxyz = np.array(coords[3:], dtype=np.float64)
    Rmat = R.from_euler(convention, rxyz, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = xyz
    return T


def solve_handeye_one(R_g2b, t_g2b, R_p2c, t_p2c, method):
    R_b2g = [Rg.T for Rg in R_g2b]
    t_b2g = [-Rg.T @ tg for Rg, tg in zip(R_g2b, t_g2b)]
    Rc, tc = cv2.calibrateHandEye(R_b2g, t_b2g, R_p2c, t_p2c, method=method)
    T = np.eye(4)
    T[:3, :3] = Rc
    T[:3, 3] = tc.reshape(3)
    in_grip = []
    for Rg, tg, Rp, tp in zip(R_g2b, t_g2b, R_p2c, t_p2c):
        p_base = T[:3, :3] @ tp + T[:3, 3]
        p_grip = Rg.T @ (p_base - tg)
        in_grip.append(p_grip)
    in_grip = np.array(in_grip)
    residual = float(np.linalg.norm(in_grip - in_grip.mean(axis=0), axis=1).mean())
    return T, residual


def solve_all_conventions(coords_all, R_p2c_all, t_p2c_all):
    """Try every Euler convention × every hand-eye method. Report the winner."""
    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    best = None  # (convention, method, T, residual)
    print(f"\n{'convention':12s}  {'method':12s}  {'cam_xyz':30s}  residual")
    print("-" * 80)
    for conv in EULER_CONVENTIONS:
        try:
            Tg = [coords_to_T(c, conv) for c in coords_all]
            R_g2b = [T[:3, :3] for T in Tg]
            t_g2b = [T[:3, 3] for T in Tg]
            for mname, mid in methods.items():
                T, res = solve_handeye_one(R_g2b, t_g2b, R_p2c_all, t_p2c_all, mid)
                cam = T[:3, 3]
                cam_str = f"({cam[0]:7.1f},{cam[1]:7.1f},{cam[2]:7.1f})"
                mark = ""
                if best is None or res < best[3]:
                    best = (conv, mname, T, res)
                    mark = "  ← best so far"
                if res < 50:  # skip the obviously broken combos to keep output readable
                    print(f"{conv:12s}  {mname:12s}  {cam_str:30s}  {res:6.2f}{mark}")
        except Exception as e:
            print(f"{conv:12s}  (failed: {e})")
    return best


def save_calib(T_cam2base, K, dist, convention, method, residual, n_poses):
    out = {
        "mode": "eye_to_hand_pattern_apionly",
        "camera_matrix": K.tolist(),
        "dist_coeffs": [dist.tolist()],
        "image_size": [FRAME_W, FRAME_H],
        "T_cam2base": T_cam2base.tolist(),
        "handeye_method": method,
        "handeye_residual_mm": residual,
        "handeye_pose_count": n_poses,
        "euler_convention": convention,
        "chessboard": {"inner_cols": BOARD_COLS, "inner_rows": BOARD_ROWS,
                       "square_mm": SQUARE_MM},
        "note": "Uses mc.get_coords() (firmware FK), no ikpy/URDF.",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


def main():
    print("Loading intrinsics...")
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
    print("This script uses mc.get_coords() for gripper pose (no ikpy/URDF).")
    print("It will try all 12 Euler conventions and pick the best.")
    print()

    coords_all = []
    R_p2c_all, t_p2c_all = [], []
    cv2.namedWindow("handeye_api", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("handeye_api", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_board(gray)

        disp = frame.copy()
        if ok:
            cv2.drawChessboardCorners(disp, (BOARD_COLS, BOARD_ROWS), corners, ok)
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.putText(disp,
                    f"poses: {len(coords_all)}    "
                    f"board: {'DETECTED' if ok else 'no'}    "
                    f"SPACE=cap  U=undo  ENTER=solve  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("handeye_api", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Quitting without solve.")
            cap.release(); cv2.destroyAllWindows()
            return
        if k == ord('u'):
            if coords_all:
                coords_all.pop()
                R_p2c_all.pop(); t_p2c_all.pop()
                print(f"  ← undo  (now {len(coords_all)} poses)")
            continue
        if k == 32:  # SPACE
            if not ok:
                print("  ✗ no chessboard in view")
                continue
            # Flush stale camera frames, bracket coord reads.
            for _ in range(3):
                cap.grab()
            c1 = read_coords_stable(mc)
            ret2, fresh = cap.read()
            c2 = read_coords_stable(mc)
            if not ret2 or c1 is None or c2 is None:
                print(f"  ✗ bad read  c1={c1}  c2={c2}  cam_ok={ret2}")
                continue
            drift_xyz = max(abs(a - b) for a, b in zip(c1[:3], c2[:3]))
            drift_rxyz = max(abs(a - b) for a, b in zip(c1[3:], c2[3:]))
            if drift_xyz > 3.0 or drift_rxyz > 2.0:
                print(f"  ✗ arm wobbled  Δxyz={drift_xyz:.1f}mm  Δrxyz={drift_rxyz:.1f}°")
                continue
            coords_avg = [(a + b) / 2.0 for a, b in zip(c1, c2)]
            gray2 = cv2.cvtColor(fresh, cv2.COLOR_BGR2GRAY)
            ok2, corners2 = detect_board(gray2)
            if not ok2:
                print("  ✗ chessboard not in synced frame")
                continue
            Rp, tp = pattern_pose(corners2, K, dist)
            if Rp is None:
                print("  ✗ PnP failed")
                continue
            coords_all.append(coords_avg)
            R_p2c_all.append(Rp)
            t_p2c_all.append(tp)
            print(f"  ✓ pose {len(coords_all):2d}  "
                  f"xyz=({coords_avg[0]:6.1f},{coords_avg[1]:6.1f},{coords_avg[2]:6.1f})  "
                  f"rxyz=({coords_avg[3]:+6.1f},{coords_avg[4]:+6.1f},{coords_avg[5]:+6.1f})")
            continue
        if k in (13, 10):  # ENTER
            if len(coords_all) < MIN_POSES:
                print(f"  need >= {MIN_POSES} poses, have {len(coords_all)}")
                continue
            print(f"\nSolving with {len(coords_all)} poses, trying all Euler conventions...")
            best = solve_all_conventions(coords_all, R_p2c_all, t_p2c_all)
            if best is None:
                print("All conventions failed to solve.")
                break
            conv, mname, T, res = best
            cam = T[:3, 3]
            print(f"\n*** BEST ***")
            print(f"  convention: {conv}  (mycobot's actual Euler convention)")
            print(f"  method:     {mname}")
            print(f"  residual:   {res:.2f} mm")
            print(f"  camera in base: ({cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f})")

            if os.path.exists(EXISTING_CALIB_PATH):
                with open(EXISTING_CALIB_PATH) as f: old = json.load(f)
                if "T_cam2base" in old:
                    old_cam = np.array(old["T_cam2base"])[:3, 3]
                    delta = np.linalg.norm(cam - old_cam)
                    print(f"  delta vs ikpy-based calibration: {delta:.1f} mm")
                    old_res = old.get("handeye_residual_mm", "?")
                    print(f"  ikpy residual was: {old_res}")

            save_calib(T, K, dist, conv, mname, res, len(coords_all))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
