"""
Eye-in-hand calibration.
Camera mounted on the end-effector. ChArUco board fixed on the desk.
Solves for T_cam2gripper (transform from camera frame to flange frame).

Uses mc.get_coords() (firmware FK, xyz extrinsic Euler) for robot pose —
NOT ikpy. That means the calibration result works with send_coords at runtime.

Workflow:
  1. Place ChArUco board flat on the desk (fixed).
  2. Run script. Servos release; drag arm by hand.
  3. Position arm so the camera sees the WHOLE board, hold steady, press SPACE.
  4. Repeat for 15-20 poses. Vary rotation on all 3 axes (twist wrist, tilt).
  5. ENTER to solve and save.
"""
import json
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280

# ── config ──────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0   # USB webcam (on-arm)
FRAME_W, FRAME_H = 1920, 1080

# ChArUco: 5 wide × 7 tall, 30 mm squares, 22 mm markers, DICT_4X4_50
BOARD_SQUARES_X = 5
BOARD_SQUARES_Y = 7
SQUARE_MM = 30.0
MARKER_MM = 22.0
DICTIONARY = cv2.aruco.DICT_4X4_50

INTRINSICS_PATH = "/Users/v/local-ai-robot-arm/gantry_calib/intrinsics.json"
OUT_PATH = "/Users/v/local-ai-robot-arm/calibration_eyeinhand.json"

MIN_POSES = 8


def load_intrinsics():
    with open(INTRINSICS_PATH) as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1)
    return K, dist


def coords_to_T(coords):
    """[x,y,z,rx,ry,rz] (mm, deg) → 4x4 flange-to-base. xyz extrinsic Euler."""
    x, y, z, rx, ry, rz = coords
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T


def make_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        SQUARE_MM, MARKER_MM, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, aruco_dict, detector


def detect_board_pose(frame, board, detector, K, dist):
    """Return (R, t, corners) — board pose in camera frame, or (None, None, None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None, None, None
    obj_points, img_points = board.matchImagePoints(corners, ids)
    if obj_points is None or len(obj_points) < 6:
        return None, None, None
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    except cv2.error:
        return None, None, None
    if not ok:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3), corners


def _rotation_angle_deg(R):
    """Rotation matrix → angle magnitude in degrees (Rodrigues angle)."""
    # angle = arccos((trace(R) - 1) / 2), clipped for numerical safety
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def solve_all_methods(R_g2b, t_g2b, R_t2c, t_t2c):
    """Try 4 hand-eye solvers, return the best (lowest board-in-base residual).

    Reports BOTH:
      - trans_res_mm: how much the board's estimated position wanders across poses
      - rot_res_deg:  how much the board's estimated orientation wanders across poses
    """
    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    best = None
    print(f"  {'method':10s}  T_cam2gripper (mm)          trans_res_mm  rot_res_deg")
    for name, m in methods.items():
        Rc2g, tc2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
        T = np.eye(4)
        T[:3, :3] = Rc2g
        T[:3, 3] = tc2g.reshape(3)
        # Board is fixed; its 6-DOF pose in BASE should be constant across poses.
        board_pos = []
        board_rot = []
        for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
            T_g2b = np.eye(4); T_g2b[:3, :3] = Rg; T_g2b[:3, 3] = tg
            T_t2c = np.eye(4); T_t2c[:3, :3] = Rt; T_t2c[:3, 3] = tt
            T_t2b = T_g2b @ T @ T_t2c
            board_pos.append(T_t2b[:3, 3])
            board_rot.append(T_t2b[:3, :3])
        pos = np.array(board_pos)
        trans_res = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())

        # Rotation residual: for each pose, angle between that R and the mean R.
        # "Mean" rotation via SVD projection of averaged rotation matrices.
        R_mean_raw = np.mean(np.stack(board_rot), axis=0)
        U, _, Vt = np.linalg.svd(R_mean_raw)
        R_mean = U @ Vt
        if np.linalg.det(R_mean) < 0:
            U[:, -1] *= -1
            R_mean = U @ Vt
        rot_errs = [_rotation_angle_deg(R.T @ R_mean) for R in board_rot]
        rot_res = float(np.mean(rot_errs))

        print(f"  {name:10s}  ({T[0,3]:+6.1f},{T[1,3]:+6.1f},{T[2,3]:+6.1f})  "
              f"       {trans_res:6.2f}       {rot_res:5.2f}°")
        if best is None or trans_res < best[2]:
            best = (name, T, trans_res, rot_res)
    return best


def main():
    K, dist = load_intrinsics()
    print(f"Loaded intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    board, aruco_dict, detector = make_board()
    print(f"ChArUco: {BOARD_SQUARES_X}x{BOARD_SQUARES_Y}, {SQUARE_MM}mm squares, {MARKER_MM}mm markers")

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    print("Warming up camera (5s)...")
    t0 = time.time()
    warm_count = 0
    while time.time() - t0 < 5.0:
        ok, _ = cap.read()
        if ok:
            warm_count += 1
    print(f"  camera warmup: {warm_count} good frames in 5s")
    if warm_count == 0:
        print("  ✗ camera never returned a frame.")
        print("     Try: unplug/replug the USB webcam, disable iPhone Continuity Camera")
        print("     (System Settings → General → AirDrop & Handoff → Continuity Camera OFF)")
        return

    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(1.5)
    mc.release_all_servos()
    print("Servos released. Move the arm by hand.\n")
    print(f"Aim for {MIN_POSES}+ poses with rotation diversity.")
    print("Keys: SPACE=capture  U=undo  ENTER=solve  Q=quit\n")

    cv2.namedWindow("eyeinhand", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("eyeinhand", 1280, 720)

    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    frame_count = 0
    failed_reads = 0
    last_heartbeat = time.time()
    while True:
        ret, frame = cap.read()
        if time.time() - last_heartbeat > 2.0:
            print(f"  [heartbeat] loop alive — good frames: {frame_count}, failed reads: {failed_reads}")
            last_heartbeat = time.time()
        if not ret:
            failed_reads += 1
            time.sleep(0.02); continue
        frame_count += 1
        R, t, corners = detect_board_pose(frame, board, detector, K, dist)
        disp = frame.copy()
        detected = R is not None
        if detected:
            rvec, _ = cv2.Rodrigues(R)
            cv2.drawFrameAxes(disp, K, dist, rvec, t, 50)
            if corners is not None:
                for c in corners:
                    x, y = int(c[0][0]), int(c[0][1])
                    cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
        color = (0, 200, 0) if detected else (0, 0, 255)
        cv2.putText(disp,
                    f"poses: {len(R_g2b_list)}   board: {'DETECTED' if detected else 'no'}   "
                    f"SPACE=cap  U=undo  ENTER=solve  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("eyeinhand", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Quit without solve.")
            cap.release(); cv2.destroyAllWindows(); return
        if k == ord('u') and R_g2b_list:
            R_g2b_list.pop(); t_g2b_list.pop()
            R_t2c_list.pop(); t_t2c_list.pop()
            print(f"  ← undo (now {len(R_g2b_list)} poses)")
            continue
        if k == 32:  # SPACE
            if not detected:
                print("  ✗ board not detected")
                continue
            # Flush camera buffer, bracket coord reads.
            for _ in range(3): cap.grab()
            c1 = mc.get_coords()
            ret2, fresh = cap.read()
            c2 = mc.get_coords()
            if (not isinstance(c1, (list, tuple)) or not isinstance(c2, (list, tuple))
                    or len(c1) != 6 or len(c2) != 6):
                print(f"  ✗ bad get_coords: c1={c1} c2={c2}"); continue
            drift = max(abs(a - b) for a, b in zip(c1, c2))
            if drift > 3.0:
                print(f"  ✗ arm wobbled ({drift:.1f}) — hold steadier"); continue
            coords_avg = [(a + b) / 2 for a, b in zip(c1, c2)]
            # Re-detect on fresh (synced) frame
            R2, t2, _ = detect_board_pose(fresh, board, detector, K, dist)
            if R2 is None:
                print("  ✗ board not detected in synced frame"); continue
            T_g2b = coords_to_T(coords_avg)
            R_g2b_list.append(T_g2b[:3, :3])
            t_g2b_list.append(T_g2b[:3, 3])
            R_t2c_list.append(R2)
            t_t2c_list.append(t2)
            print(f"  ✓ pose {len(R_g2b_list):2d}  "
                  f"flange=({coords_avg[0]:6.1f},{coords_avg[1]:6.1f},{coords_avg[2]:6.1f}) "
                  f"eul=({coords_avg[3]:+6.1f},{coords_avg[4]:+6.1f},{coords_avg[5]:+6.1f})  "
                  f"board_cam=({t2[0]:6.1f},{t2[1]:6.1f},{t2[2]:6.1f})")
            continue
        if k in (13, 10):  # ENTER
            if len(R_g2b_list) < MIN_POSES:
                print(f"  need ≥ {MIN_POSES} poses, have {len(R_g2b_list)}")
                continue
            print(f"\nSolving with {len(R_g2b_list)} poses...")
            name, T, res, rot_res = solve_all_methods(
                R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list)
            print(f"\nBest: {name}  trans_res={res:.2f}mm  rot_res={rot_res:.2f}°")
            print(f"T_cam2gripper translation: ({T[0,3]:.1f}, {T[1,3]:.1f}, {T[2,3]:.1f}) mm")
            # Predict runtime XY error at ~200mm workspace reach
            predicted_edge_err = 200.0 * np.tan(np.radians(rot_res))
            print(f"Estimated XY error at 200mm reach: ~{predicted_edge_err:.1f}mm (from {rot_res:.2f}° rotation error)")

            out = {
                "mode": "eye_in_hand",
                "camera_matrix": K.tolist(),
                "dist_coeffs": [dist.tolist()],
                "image_size": [FRAME_W, FRAME_H],
                "T_cam2gripper": T.tolist(),
                "handeye_method": name,
                "handeye_residual_mm": res,
                "handeye_rotation_residual_deg": rot_res,
                "handeye_pose_count": len(R_g2b_list),
                "charuco": {
                    "squares_x": BOARD_SQUARES_X, "squares_y": BOARD_SQUARES_Y,
                    "square_mm": SQUARE_MM, "marker_mm": MARKER_MM,
                    "dictionary": "DICT_4X4_50",
                },
                "note": ("Eye-in-hand calibration. Robot pose from mc.get_coords() "
                         "(xyz extrinsic Euler). Use with send_coords, not ikpy."),
            }
            with open(OUT_PATH, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Wrote {OUT_PATH}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
