"""
Live ArUco tracking — pump tip hovers above marker id 2 and follows it
in XY as you slide the marker around the desktop.

Verifies hand-eye calibration end-to-end under continuous motion.

Keys (in preview window):
  Q     = stop tracking, return home, quit
  SPACE = (optional) descend briefly to 5mm above marker, then lift back

Throttle: only sends new IK if marker has moved > MOVE_THRESHOLD_MM
or every UPDATE_PERIOD_S seconds.
"""
import json
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280
from ikpy.chain import Chain

SERIAL_PORT = "/dev/tty.usbserial-54780106801"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
URDF_PATH = "/Users/v/Downloads/69conference/mycobot_280_m5.urdf"
CALIB_PATH = "/Users/v/Downloads/69conference/calibration_result.json"

MARKER_ID = 2
ARUCO_DICT = cv2.aruco.DICT_6X6_50
PUMP_LENGTH = 70.0
HOVER_HEIGHT = 50.0       # mm pump-tip above marker during tracking
DIP_HEIGHT = 5.0          # mm pump-tip above marker during SPACE-dip
SPEED = 25
MOVE_THRESHOLD_MM = 5.0   # don't bother updating if target moved less than this
UPDATE_PERIOD_S = 0.25    # at most this often even if marker is moving fast
MAX_JOINT_STEP_DEG = 90   # safety: refuse big wrist flips during tracking


def load_calib():
    with open(CALIB_PATH) as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1)
    T = np.array(d["T_cam2base"], dtype=np.float64)
    return K, dist, T, d


def detect_marker(frame, target_id):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(d, p)
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None:
        return None, None, []
    ids_flat = ids.flatten().tolist()
    for i, mid in enumerate(ids_flat):
        if int(mid) == target_id:
            return corners[i][0].mean(axis=0), corners[i][0], ids_flat
    return None, None, ids_flat


def pixel_to_base_z0(u, v, K, dist, T_cam2base):
    pts = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray_cam = np.array([(und[0] - cx) / fx, (und[1] - cy) / fy, 1.0])
    cam_origin = T_cam2base[:3, 3]
    ray_base = T_cam2base[:3, :3] @ ray_cam
    if abs(ray_base[2]) < 1e-9:
        return None
    t = -cam_origin[2] / ray_base[2]
    if t <= 0:
        return None
    return cam_origin + t * ray_base


chain = Chain.from_urdf_file(
    URDF_PATH, base_elements=['g_base'], last_link_vector=[0, 0, 0],
    active_links_mask=[False, False, True, True, True, True, True, True, False],
)


def solve_ik(target_xyz_mm, current_angles_deg):
    target_T = np.eye(4)
    target_T[:3, 3] = np.array(target_xyz_mm) / 1000.0
    target_T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    init = [0.0] * len(chain.links)
    for i, deg in enumerate(current_angles_deg):
        init[i + 2] = float(np.radians(deg))
    j = chain.inverse_kinematics_frame(target_T, initial_position=init, orientation_mode="Z")
    return [float(np.degrees(j[i + 2])) for i in range(6)]


def fk_pos_mm(angles_deg):
    pose = [0.0] * len(chain.links)
    for i, deg in enumerate(angles_deg):
        pose[i + 2] = float(np.radians(deg))
    return chain.forward_kinematics(pose)[:3, 3] * 1000.0


def try_move(mc, target_xyz_mm, cur_angles, max_delta_deg):
    """Returns (joints, fk_err_mm) on success, or (None, reason) on reject."""
    try:
        joints = solve_ik(target_xyz_mm, cur_angles)
    except Exception as e:
        return None, f"IK exception: {e}"
    fk = fk_pos_mm(joints)
    err = float(np.linalg.norm(np.array(fk) - np.array(target_xyz_mm)))
    if err > 15:
        return None, f"IK error too large: {err:.1f}mm"
    delta = max(abs(j - c) for j, c in zip(joints, cur_angles))
    if delta > max_delta_deg:
        return None, f"joint swing too large: {delta:.0f}° (limit {max_delta_deg}°)"
    mc.send_angles(joints, SPEED)
    return joints, err


def main():
    K, dist, T_cam2base, calib = load_calib()
    print(f"Calibration: mode={calib.get('mode')}, residual={calib.get('handeye_residual_mm', '?')} mm")
    print(f"Camera in base: ({T_cam2base[0,3]:.1f}, {T_cam2base[1,3]:.1f}, {T_cam2base[2,3]:.1f})")
    print()

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8):
        cap.read()

    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(1.0)
    print("Going home (0,0,0,0,0,0)...")
    mc.send_angles([0, 0, 0, 0, 0, 0], SPEED)
    time.sleep(3.5)

    print(f"Tracking marker id={MARKER_ID}. Move the marker around the desk.")
    print(f"  Q = quit (returns home first)")
    print(f"  SPACE = dip pump tip to {DIP_HEIGHT}mm above marker, then lift")
    print()

    cv2.namedWindow("aruco_track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aruco_track", 1280, 720)

    last_target = None
    last_move_t = 0.0
    last_print_t = 0.0
    dip_requested = False
    first_move_done = False  # first move from home needs a big swing budget

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue

        pixel, corners, all_ids = detect_marker(frame, MARKER_ID)
        disp = frame.copy()

        if pixel is not None:
            for i in range(4):
                p1 = tuple(int(x) for x in corners[i])
                p2 = tuple(int(x) for x in corners[(i + 1) % 4])
                cv2.line(disp, p1, p2, (0, 255, 0), 2)
            cx, cy = int(pixel[0]), int(pixel[1])
            cv2.circle(disp, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(disp, f"id {MARKER_ID}", (cx + 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        status = (f"marker {'OK' if pixel is not None else 'lost'}    "
                  f"Q=quit  SPACE=dip")
        cv2.putText(disp, status, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if pixel is not None else (0, 0, 255), 2)
        cv2.imshow("aruco_track", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == 32 and pixel is not None:
            dip_requested = True

        if pixel is None:
            continue

        base_xyz = pixel_to_base_z0(pixel[0], pixel[1], K, dist, T_cam2base)
        if base_xyz is None:
            continue

        now = time.time()
        z_target = PUMP_LENGTH + (DIP_HEIGHT if dip_requested else HOVER_HEIGHT)
        target = [float(base_xyz[0]), float(base_xyz[1]), z_target]

        moved_enough = (last_target is None or
                        np.linalg.norm(np.array(target) - np.array(last_target)) > MOVE_THRESHOLD_MM)
        time_elapsed = (now - last_move_t) > UPDATE_PERIOD_S

        if not (moved_enough or dip_requested):
            continue
        if not time_elapsed and not dip_requested:
            continue

        cur = mc.get_angles()
        if not cur or len(cur) != 6:
            continue

        was_first_move = not first_move_done
        max_delta = 180 if was_first_move else MAX_JOINT_STEP_DEG
        joints, info = try_move(mc, target, cur, max_delta)
        if joints is None:
            if now - last_print_t > 1.0:
                print(f"  ✗ {info}")
                last_print_t = now
            continue

        if was_first_move:
            print("  → first big move sent — settling 3s before tracking starts...")
            time.sleep(3.0)

        first_move_done = True
        last_target = target
        last_move_t = now
        if now - last_print_t > 0.5:
            print(f"  → marker_base=({base_xyz[0]:6.1f},{base_xyz[1]:6.1f})  "
                  f"target_z={z_target:.0f}  ik_err={info:.1f}mm")
            last_print_t = now

        if dip_requested:
            # Hold the dip briefly so you can see how close it landed
            time.sleep(1.5)
            dip_requested = False
            last_target = None  # force re-issue at hover height

    print("\nReturning home...")
    mc.send_angles([0, 0, 0, 0, 0, 0], SPEED)
    time.sleep(3.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
