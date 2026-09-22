"""
Hover-on-cube test using simple HSV color detection + 2D affine.

No OWL, no LLM. Just HSV thresholding for a colored blob on the desk.

Keys (in preview window):
  SPACE      = hover pump tip above the (stable) cube
  H          = go home (arm out of camera view)
  1..5       = switch color (green / blue / pink / yellow / red)
  M          = toggle mask view (see exactly what colour is matched)
  Q          = quit
  LEFT-CLICK = resample the cube's colour under current lighting

Fixes vs the original: camera auto-exposure/white-balance locked so the colour
stops drifting; click-to-sample colour; median-smoothed center so the box stops
jumping; and the firmware tool frame is set so send_coords drives the PUMP TIP
(pump-aware) instead of the bare flange.
"""
import json
import time
from collections import deque
import numpy as np
import cv2
from pymycobot.mycobot280 import MyCobot280

SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080

CALIB_PATH = "/Users/v/local-ai-robot-arm/calibration_affine2d.json"

TARGET_COLOR = "green"   # start on green

# HSV starting ranges. Better: CLICK the cube in the window to resample its
# real color under your lighting (fixes "doesn't recognize"). Hue 0-179.
COLOR_RANGES = {
    "green":  [(35, 80, 60), (85, 255, 255)],
    "blue":   [(95, 80, 60), (135, 255, 255)],
    "pink":   [(140, 60, 100), (175, 255, 255)],
    "yellow": [(18, 80, 80), (35, 255, 255)],
    "red":    [(0, 100, 80), (10, 255, 255)],   # low-red band only
}
COLOR_KEYS = list(COLOR_RANGES.keys())

MIN_AREA_PX = 500     # ignore blobs smaller than this

PUMP_LENGTH = 70.0    # mm: pump tip beyond the flange along tool +Z
# NOTE: with the tool frame set (set_end_type(1)), HOVER_Z_MM is the PUMP TIP
# height, not the flange. Tune it live in the window with -/+ , then set the
# value you like here. Still conservative — lower with '-' toward the cube.
HOVER_Z_MM = 80.0
DOWN_ORIENTATION = (180.0, 0.0, 0.0)
SPEED = 30
HOME_ANGLES = [0, 0, 0, 0, 0, 0]

# Detection smoothing + camera lock (stops the box jumping / color drifting)
SMOOTH_N = 5             # frames of history for median-smoothed center
STABLE_RADIUS_PX = 25    # center must stay within this to count as "stable"
HSV_H_TOL = 12           # hue tolerance when you click-sample a color
LOCK_EXPOSURE = True     # try to disable camera auto-exposure / auto-WB


def load_affine():
    with open(CALIB_PATH) as f:
        d = json.load(f)
    A = np.array(d["affine_2x3"], dtype=np.float64)
    return A, d


def pixel_to_base_xy(u, v, A):
    p = np.array([u, v, 1.0])
    xy = A @ p
    return float(xy[0]), float(xy[1])


SETTLE_TOL_MM = 1.0      # movement below this counts as "not moving"
SETTLE_STABLE_READS = 3  # consecutive stable reads before we call it settled
SETTLE_MIN_S = 0.8       # never declare settled before this (arm may not have started)


def wait_settled(mc, timeout=15.0):
    """Wait for a move to finish, then return the final coords.

    Does NOT use is_moving() — that flag is unreliable on this arm (it can
    read 0 before motion starts and flickers mid-move). Instead we poll the
    actual position and declare the move finished once it stops changing.
    """
    t0 = time.time()
    last, stable = None, 0
    while time.time() - t0 < timeout:
        try:
            c = mc.get_coords()
        except Exception:
            c = None
        if isinstance(c, (list, tuple)) and len(c) == 6:
            if last is not None:
                d = sum((a - b) ** 2 for a, b in zip(c[:3], last[:3])) ** 0.5
                stable = stable + 1 if d < SETTLE_TOL_MM else 0
            last = c
            if stable >= SETTLE_STABLE_READS and (time.time() - t0) >= SETTLE_MIN_S:
                break
        time.sleep(0.15)
    time.sleep(0.2)
    return last


def lock_camera(cap):
    """Best-effort: turn OFF auto-exposure and auto-white-balance so the cube's
    colour stops drifting in/out of range. macOS/AVFoundation honours only some
    of these — any that fail are harmless."""
    if not LOCK_EXPOSURE:
        return
    for prop, val, name in [
        (cv2.CAP_PROP_AUTO_WB, 0, "auto-WB off"),
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "manual exposure"),  # 0.25=manual on many
    ]:
        try:
            cap.set(prop, val)
        except Exception:
            pass
    for _ in range(10):
        cap.read()


def make_mouse_cb(state):
    """Left-click the cube to resample TARGET_COLOR's HSV at the current
    exposure — robust to whatever the lighting actually is."""
    def cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        frame = state.get("frame")
        if frame is None:
            return
        h_img, w_img = frame.shape[:2]
        if not (0 <= x < w_img and 0 <= y < h_img):
            return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = [int(z) for z in hsv[y, x]]
        lo = (max(h - HSV_H_TOL, 0), max(s - 70, 40), max(v - 70, 40))
        hi = (min(h + HSV_H_TOL, 179), 255, 255)
        COLOR_RANGES[TARGET_COLOR] = [lo, hi]
        state["msg"] = f"sampled {TARGET_COLOR}: H{h} S{s} V{v}"
        print(f"  ✓ sampled {TARGET_COLOR} HSV=({h},{s},{v}) -> {lo}..{hi}")
    return cb


def detect_color_blob(frame, color):
    """Return ((cx, cy), area, bbox) of the largest blob matching color, or None."""
    lo, hi = COLOR_RANGES[color]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_AREA_PX:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    x, y, w, h = cv2.boundingRect(best)
    return ((cx, cy), area, (x, y, x + w, y + h))


def main():
    global TARGET_COLOR

    A, calib = load_affine()
    print(f"Calibration: {calib.get('mode')}  residuals: {calib.get('residuals_mm')}")
    print(f"Hover Z: {HOVER_Z_MM} mm above base origin\n")

    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(10): cap.read()
    lock_camera(cap)   # disable auto-exposure / auto-WB so colour stays stable

    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(1.5)
    try: mc.power_on(); time.sleep(0.5)
    except Exception: pass
    try: mc.set_free_mode(0)
    except Exception: pass
    for sid in range(1, 7):
        try: mc.focus_servo(sid)
        except Exception: pass
    time.sleep(0.3)

    # Tell the firmware the pump is mounted, so send_coords drives the PUMP TIP
    # (not the bare flange) and keeps the pump pointing down — stops it from
    # missing the cube by the pump length and from folding into the arm.
    try:
        mc.set_tool_reference([0, 0, PUMP_LENGTH, 0, 0, 0])
        mc.set_end_type(1)   # 1 = tool frame
        print(f"  tool frame set: pump tip = flange + {PUMP_LENGTH:.0f}mm")
    except Exception as e:
        print(f"  ⚠ could not set tool frame ({e}); IK won't know about the pump")

    print("Going home (out of camera view)...")
    mc.send_angles(HOME_ANGLES, SPEED)
    wait_settled(mc, 6.0)
    print(f"Ready. Target color: {TARGET_COLOR}\n")
    print("Keys: SPACE=hover  H=home  1..5=color  M=mask view  Q=quit")
    print("Tip: LEFT-CLICK the cube to resample its colour if not detected.\n")

    cv2.namedWindow("cube_hover", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("cube_hover", 1280, 720)
    state = {"frame": None, "msg": ""}
    cv2.setMouseCallback("cube_hover", make_mouse_cb(state))

    centers = deque(maxlen=SMOOTH_N)   # recent raw centers for smoothing
    show_mask = False
    hover_z = HOVER_Z_MM               # live-adjustable pump-tip hover height
    last_xy = None                     # last hovered XY, so -/+ can re-move in place

    def smoothed():
        """Return (cx, cy, stable) from the recent-center history, or None."""
        if len(centers) < SMOOTH_N:
            return None
        arr = np.array(centers, dtype=np.float64)
        c = np.median(arr, axis=0)
        spread = float(np.max(np.linalg.norm(arr - c, axis=1)))
        return c[0], c[1], (spread <= STABLE_RADIUS_PX)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        state["frame"] = frame

        det = detect_color_blob(frame, TARGET_COLOR)
        if det is not None:
            centers.append(det[0])
        else:
            centers.clear()
        sm = smoothed()

        if show_mask:
            lo, hi = COLOR_RANGES[TARGET_COLOR]
            m = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                            np.array(lo), np.array(hi))
            disp = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        else:
            disp = frame.copy()

        if det is not None:
            (cx, cy), area, (x1, y1, x2, y2) = det
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 0), 2)
        # smoothed, stability-aware marker is what we actually hover on
        if sm is not None:
            scx, scy, stable = sm
            col = (0, 255, 0) if stable else (0, 165, 255)
            cv2.circle(disp, (int(scx), int(scy)), 7, col, -1)
            cv2.putText(disp, "STABLE" if stable else "settling...",
                        (int(scx) + 10, int(scy)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, 2)

        found_txt = "FOUND" if det else "not found"
        hdr_col = (0, 200, 0) if det is not None else (0, 0, 255)
        cv2.putText(disp,
                    f"target: {TARGET_COLOR}  {found_txt}   tipZ={hover_z:.0f}mm   "
                    f"SPACE=hover  -/+=lower/raise  H=home  1-5=color  M=mask  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hdr_col, 2)
        if state["msg"]:
            cv2.putText(disp, state["msg"], (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("cube_hover", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('m'):
            show_mask = not show_mask
            continue
        if k in (ord('-'), ord('_'), ord('='), ord('+')):
            # -/_ lower, =/+ raise. Move the arm IN PLACE right away if we've
            # already hovered somewhere (so you see it drop/raise live).
            if k in (ord('-'), ord('_')):
                hover_z = max(hover_z - 10, 10)
            else:
                hover_z = min(hover_z + 10, 250)
            state["msg"] = f"tip hover Z = {hover_z:.0f}mm"
            print(f"  tip hover Z = {hover_z:.0f}mm")
            if last_xy is not None:
                rx, ry, rz = DOWN_ORIENTATION
                mc.send_coords([last_xy[0], last_xy[1], hover_z, rx, ry, rz], SPEED, 0)
                wait_settled(mc, 10.0)
            else:
                print("    (press SPACE on the cube first, then -/+ move it live)")
            continue
        if k == ord('h'):
            print("\n[H] returning home...")
            mc.send_angles(HOME_ANGLES, SPEED)
            wait_settled(mc, 6.0)
            state["msg"] = "at home"
            continue
        if k in (ord(str(i + 1)) for i in range(len(COLOR_KEYS))):
            TARGET_COLOR = COLOR_KEYS[int(chr(k)) - 1]
            centers.clear()
            print(f"\ntarget color → {TARGET_COLOR}")
            state["msg"] = f"target: {TARGET_COLOR}"
            continue
        if k == 32:  # SPACE — hover on the stable smoothed center
            if sm is None:
                print(f"\n[SPACE] no stable {TARGET_COLOR} cube yet — "
                      "hold still / click the cube to resample")
                state["msg"] = f"no stable {TARGET_COLOR} cube"
                continue
            scx, scy, stable = sm
            if not stable:
                print("\n[SPACE] cube not stable yet — wait for green STABLE marker")
                state["msg"] = "not stable — wait"
                continue
            x, y = pixel_to_base_xy(scx, scy, A)
            last_xy = (x, y)
            rx, ry, rz = DOWN_ORIENTATION
            target = [x, y, hover_z, rx, ry, rz]
            print(f"\n[SPACE] {TARGET_COLOR}  pixel=({scx:.0f},{scy:.0f})")
            print(f"  tip target base_xy=({x:.1f}, {y:.1f})  tip Z={hover_z:.0f}")
            mc.send_coords(target, SPEED, 0)
            wait_settled(mc, 15.0)
            actual = mc.get_coords()
            if isinstance(actual, (list, tuple)) and len(actual) == 6:
                dx = actual[0] - x
                dy = actual[1] - y
                print(f"  arrived=({actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f})  "
                      f"XY err=(dx={dx:+.1f}, dy={dy:+.1f})")
            state["msg"] = "hover sent — H to home"

    print("\nQuitting — returning home...")
    try: mc.set_end_type(0)   # restore flange frame for other scripts
    except Exception: pass
    mc.send_angles(HOME_ANGLES, SPEED)
    wait_settled(mc, 6.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
