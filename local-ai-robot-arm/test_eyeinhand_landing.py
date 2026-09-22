"""
Eye-in-hand accuracy test — interactive loop with live preview.

Workflow:
  1. Arm goes to observe pose (looking down at desk).
  2. Live camera window shows detected markers.
  3. Press SPACE to grab a frame + robot pose, compute marker in base,
     send arm to hover above it.
  4. After landing, press R to return to observe pose.
  5. Move marker to a new spot, press SPACE again. Repeat.
  6. Q to quit.
"""
import json
import time
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280

SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080

CALIB_PATH = "/Users/v/local-ai-robot-arm/calibration_eyeinhand.json"

ARUCO_DICT = cv2.aruco.DICT_6X6_50
MARKER_MM = 25.0

OBSERVE_ANGLES = [-0.17, 0.0, -18.28, -61.25, 4.65, 145.0]
HOVER_ABOVE_MM = 220
DOWN_ORIENTATION = (180.0, 0.0, 90.0)
SPEED = 25


def load_calib():
    with open(CALIB_PATH) as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).reshape(-1)
    T_c2g = np.array(d["T_cam2gripper"], dtype=np.float64)
    return K, dist, T_c2g, d


def coords_to_T(coords):
    x, y, z, rx, ry, rz = coords
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = [x, y, z]
    return T


def detect_any_marker(frame, K, dist):
    """Return (marker_id, R, t, corners) of the first marker found."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(d, p)
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None or len(ids) == 0:
        return None, None, None, None
    i = 0
    marker_id = int(ids.flatten()[i])
    half = MARKER_MM / 2.0
    obj = np.array([[-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners[i], K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return marker_id, None, None, corners[i]
    R, _ = cv2.Rodrigues(rvec)
    return marker_id, R, tvec.reshape(3), corners[i]


def wait_settled(mc, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if mc.is_moving() == 0:
                break
        except Exception:
            pass
        time.sleep(0.3)
    time.sleep(0.5)


def go_observe(mc):
    print(f"→ observe pose {OBSERVE_ANGLES}")
    mc.send_angles(OBSERVE_ANGLES, SPEED)
    wait_settled(mc, 8.0)


def do_landing(mc, K, dist, T_c2g, frame):
    """Detect marker in frame, compute base-frame pose, send landing move."""
    coords = mc.get_coords()
    if not isinstance(coords, (list, tuple)) or len(coords) != 6:
        print(f"  ✗ bad get_coords: {coords!r}")
        return None
    mid, R_m2c, t_m2c, corners = detect_any_marker(frame, K, dist)
    if mid is None:
        print("  ✗ no marker in view")
        return None
    if R_m2c is None:
        print(f"  ✗ marker id={mid} PnP failed")
        return None

    T_m2c = np.eye(4); T_m2c[:3, :3] = R_m2c; T_m2c[:3, 3] = t_m2c
    T_g2b = coords_to_T(coords)
    T_m2b = T_g2b @ T_c2g @ T_m2c
    mx, my, mz = T_m2b[:3, 3]

    tx, ty, tz = mx, my, mz + HOVER_ABOVE_MM
    rx, ry, rz = DOWN_ORIENTATION
    print(f"  marker id={mid}  cam=({t_m2c[0]:.1f},{t_m2c[1]:.1f},{t_m2c[2]:.1f})")
    print(f"  marker in base=({mx:.1f}, {my:.1f}, {mz:.1f})")
    print(f"  target={ [round(v,1) for v in [tx, ty, tz, rx, ry, rz]] }")
    mc.send_coords([tx, ty, tz, rx, ry, rz], SPEED, 0)
    wait_settled(mc, 15.0)
    actual = mc.get_coords()
    if isinstance(actual, (list, tuple)) and len(actual) == 6:
        dx = actual[0] - tx; dy = actual[1] - ty; dz = actual[2] - tz
        err = float(np.linalg.norm([dx, dy, dz]))
        print(f"  arrived=({actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f})  |Δ|={err:.1f}mm")
    return mid


def main():
    K, dist, T_c2g, calib = load_calib()
    print(f"Calibration: {calib.get('mode')}  {calib.get('handeye_method')}  "
          f"residual={calib.get('handeye_residual_mm'):.2f}mm")
    print(f"T_cam2gripper: ({T_c2g[0,3]:.1f}, {T_c2g[1,3]:.1f}, {T_c2g[2,3]:.1f}) mm")
    print(f"HOVER_ABOVE_MM = {HOVER_ABOVE_MM}\n")

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(10): cap.read()

    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(1.5)
    mc.power_on(); time.sleep(0.5)
    go_observe(mc)

    cv2.namedWindow("landing_test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("landing_test", 1280, 720)

    print("\nKeys (with camera window focused):")
    print("  SPACE = detect marker + land")
    print("  R     = return to observe pose")
    print("  Q     = quit\n")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, p)

    last_action_msg = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        corners_list, ids, _ = detector.detectMarkers(frame)
        disp = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(disp, corners_list, ids)
            n_seen = len(ids)
        else:
            n_seen = 0

        color = (0, 200, 0) if n_seen > 0 else (0, 0, 255)
        cv2.putText(disp,
                    f"markers: {n_seen}   SPACE=land  R=return  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if last_action_msg:
            cv2.putText(disp, last_action_msg, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("landing_test", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('r'):
            print("\n[R] returning to observe pose...")
            go_observe(mc)
            last_action_msg = "returned to observe"
            continue
        if k == 32:  # SPACE
            if n_seen == 0:
                print("\n[SPACE] but no marker in view — skipping")
                last_action_msg = "no marker to land on"
                continue
            print("\n[SPACE] landing on marker...")
            # Flush camera buffer to get freshest frame
            for _ in range(3): cap.grab()
            ret2, fresh = cap.read()
            if not ret2:
                continue
            mid = do_landing(mc, K, dist, T_c2g, fresh)
            last_action_msg = f"landed on id={mid} — press R to return"

    print("\nQuitting — returning to observe pose then done.")
    go_observe(mc)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
