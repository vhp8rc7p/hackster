"""
Live tracking test for the 2D affine calibration.

Hovers the pump tip above ArUco marker id 2 and follows it as you slide
the marker around the desk. Uses calibration_affine2d.json (pixel→base-XY).

This is the affine equivalent of test_aruco_landing.py — same UX,
different math under the hood.

Keys:
  Q     = quit (returns home first)
  SPACE = dip pump tip briefly to 5mm above marker
"""
import json
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280
from ikpy.chain import Chain

SERIAL_PORT = "/dev/tty.usbserial-0202EDB8"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
URDF_PATH = "/Users/v/Downloads/69conference/mycobot_280_m5.urdf"
CALIB_PATH = "/Users/v/Downloads/69conference/calibration_affine2d.json"

MARKER_ID = 2
ARUCO_DICT = cv2.aruco.DICT_6X6_50
PUMP_LENGTH = 70.0
HOVER_HEIGHT = 30.0   # lower so you can read landing accuracy without pressing SPACE
DIP_HEIGHT = -15.0    # press into surface if you want to dip (SPACE)
SPEED = 25
MOVE_THRESHOLD_MM = 5.0
UPDATE_PERIOD_S = 0.25
MAX_JOINT_STEP_DEG = 90


def load_affine():
    with open(CALIB_PATH) as f:
        d = json.load(f)
    A = np.array(d["affine_2x3"], dtype=np.float64)
    return A, d


def pixel_to_base_xy(u, v, A):
    """Apply 2D affine. Returns (x, y) in mm at the calibrated Z plane."""
    p = np.array([u, v, 1.0])
    xy = A @ p
    return float(xy[0]), float(xy[1])


def detect_marker_center(frame, target_id):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(d, p)
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None:
        return None, None
    for i, mid in enumerate(ids.flatten().tolist()):
        if int(mid) == target_id:
            return corners[i][0].mean(axis=0), corners[i][0]
    return None, None


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
    A, calib = load_affine()
    print(f"Calibration: mode={calib.get('mode')}")
    print(f"  residuals: {calib.get('residuals_mm')}")
    print(f"  table Z: {calib.get('table_z_mm')} mm")
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
    print(f"  SPACE = dip pump tip to {DIP_HEIGHT}mm above marker, then lift\n")

    cv2.namedWindow("affine_track", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("affine_track", 1280, 720)

    last_target = None
    last_move_t = 0.0
    last_print_t = 0.0
    dip_requested = False
    first_move_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue

        pixel, corners = detect_marker_center(frame, MARKER_ID)
        disp = frame.copy()
        if pixel is not None:
            for i in range(4):
                p1 = tuple(int(x) for x in corners[i])
                p2 = tuple(int(x) for x in corners[(i + 1) % 4])
                cv2.line(disp, p1, p2, (0, 255, 0), 2)
            cx, cy = int(pixel[0]), int(pixel[1])
            cv2.circle(disp, (cx, cy), 8, (0, 255, 0), -1)
        status = (f"marker {'OK' if pixel is not None else 'lost'}    "
                  f"Q=quit  SPACE=dip")
        cv2.putText(disp, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if pixel is not None else (0, 0, 255), 2)
        cv2.imshow("affine_track", disp)

        k = cv2.waitKey(1) & 0xFF
        if k != 255 and k != 0:
            print(f"  [key pressed: {k} {chr(k) if 32 <= k < 127 else '?'}]")
        if k == ord('q'):
            break
        if (k == 32 or k == ord('d')) and last_target is not None:
            # Use last-known target — arm itself occludes the marker when hovering above it
            dip_requested = True
            print(f"  ↓ DIP requested — descending on last target ({last_target[0]:.0f},{last_target[1]:.0f})")

        if pixel is None and not dip_requested:
            continue

        if pixel is not None:
            x_base, y_base = pixel_to_base_xy(pixel[0], pixel[1], A)
        else:
            # marker hidden (arm blocking) — keep using the last known XY
            x_base, y_base = last_target[0], last_target[1]

        now = time.time()
        z_target = PUMP_LENGTH + (DIP_HEIGHT if dip_requested else HOVER_HEIGHT)
        target = [x_base, y_base, z_target]

        moved_enough = (last_target is None or
                        np.linalg.norm(np.array(target) - np.array(last_target)) > MOVE_THRESHOLD_MM)
        time_elapsed = (now - last_move_t) > UPDATE_PERIOD_S
        if not (moved_enough or dip_requested) or (not time_elapsed and not dip_requested):
            continue

        cur = mc.get_angles()
        if not isinstance(cur, (list, tuple)) or len(cur) != 6:
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
            print(f"  → marker_base=({x_base:6.1f},{y_base:6.1f})  "
                  f"target_z={z_target:.0f}  ik_err={info:.1f}mm")
            last_print_t = now

        if dip_requested:
            time.sleep(1.5)
            dip_requested = False
            last_target = None

    print("\nReturning home...")
    mc.send_angles([0, 0, 0, 0, 0, 0], SPEED)
    time.sleep(3.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
