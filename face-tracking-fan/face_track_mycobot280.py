#!/usr/bin/env python3
"""
face_track_mycobot280.py

Make an Elephant Robotics myCobot 280 keep its end effector pointed at your
face, using only the laptop's built-in webcam.

Pipeline:
    webcam frame -> face pixel + apparent size -> 3D point in CAMERA frame
                 -> rigid transform -> 3D point in ROBOT BASE frame
                 -> "pointing IK" -> joint angles -> pymycobot send_angles

Frames
------
CAMERA (OpenCV convention): +x right, +y down, +z out of the lens.
BASE:   +x forward (out from the arm into the room), +y left, +z up.
        Origin on the J1 (base yaw) axis, at the mounting surface.

Joint convention assumed here: myCobot 280 zero pose = arm straight up, and a
POSITIVE J2/J3/J4 tips the arm FORWARD (toward +x). If your arm leans the wrong
way, flip PITCH_SIGN to -1. Use the 't' key to check against a known pose.

Run:  python face_track_mycobot280.py
Keys: q quit | SPACE arm/disarm motion | h home | t test pose (point forward)
      w/s trim camera pitch | a/d trim camera yaw | p print calib
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np

# ======================================================================
# 1. CONFIG - everything you need to measure/tune is in this block
# ======================================================================

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# --- Camera intrinsics (pixels). Replace with cv2.calibrateCamera() output.
# Rough starting point for a ~68 deg horizontal FOV laptop cam at 1280x720:
FX = FY = 950.0
CX, CY = FRAME_W / 2.0, FRAME_H / 2.0

# --- Monocular depth scale (Haar fallback path only).
HEAD_WIDTH_M = 0.155        # tune: sit at a measured 60 cm, adjust until Z reads 0.60
IPD_M = 0.063               # interpupillary distance, used by the FaceMesh path

# MediaPipe Tasks face model, expected next to this script.
MODEL_PATH = str(Path(__file__).resolve().parent / "face_landmarker.task")

# --- Camera pose in BASE frame -------------------------------------------------
# Tape-measure this from the arm's base axis to the webcam lens.
CAM_POS_BASE = np.array([0.35, -0.45, 0.25])   # x fwd, y left, z up  [metres]
# measured: 35 cm forward, 45 cm to the BASE's right (arm sits to the user's
# left, so the laptop is on the arm's right -> negative y). Height ASSUMED 25 cm
# above the mounting surface - re-measure if aim is off vertically.
# Nominal orientation: camera stares back along -x_base with its up along +z_base.
# Then add small trims (degrees) for how the laptop lid is actually tilted/rotated.
CAM_YAW_DEG = 0.0     # + rotates the camera's look direction toward +y_base (left)
CAM_PITCH_DEG = -8.0  # + tilts the look direction up (laptop lids usually look up a bit)
CAM_ROLL_DEG = 0.0

# --- myCobot 280 kinematics [metres] -------------------------------------------
# Nominal factory geometry. VERIFY against your own arm with a ruler - Elephant
# publishes slightly different numbers across the M5/Pi/AI revisions.
D1 = 0.13156   # mounting surface -> J2 (shoulder pitch) axis
A2 = 0.1104    # J2 -> J3  (upper arm)
A3 = 0.0960    # J3 -> J4  (forearm)
TOOL_LEN = 0.1218  # J4 wrist pitch axis -> tool tip, along the pointing axis
                   # (d5 + d6 ~= 0.0732 + 0.0486; add your gripper/pointer length)

PITCH_SIGN = +1    # flip if J2/J3 fold the wrong way

# At the 280's zero pose the arm stands vertical but the flange points
# HORIZONTALLY FORWARD - the tool axis is perpendicular to the forearm, not in
# line with it. This offset accounts for that; without it the tool aims 90 deg
# low (straight at the desk). Verified on hardware.
FLANGE_OFFSET_DEG = 90.0

# Where to park the wrist center while pointing (in the arm's vertical plane):
WRIST_RHO = 0.11   # radial distance from the J1 axis
WRIST_Z = 0.22     # height above the mounting surface

# --- Robot ---------------------------------------------------------------------
PORT = "COM10"           # CH9102 bridge, confirmed present
BAUD = 115200           # 115200 for M5/Pi over USB; 1000000 for some Pi setups
MOVE_SPEED = 50         # pymycobot speed, 1-100
START_ARMED = False     # motion disabled until you press SPACE (safety)

# id, name, min_deg, max_deg   (myCobot 280 factory limits)
JOINTS = [
    dict(name="j1_yaw",      lo=-168.0, hi=168.0),
    dict(name="j2_shoulder", lo=-135.0, hi=135.0),
    dict(name="j3_elbow",    lo=-150.0, hi=150.0),
    dict(name="j4_wrist",    lo=-145.0, hi=145.0),
    dict(name="j5_wrist2",   lo=-165.0, hi=165.0),
    dict(name="j6_roll",     lo=-180.0, hi=180.0),
]

# --- Control loop --------------------------------------------------------------
LOOP_HZ = 30            # vision rate
SEND_HZ = 10            # serial command rate (115200 will not keep up with 30)
EMA_ALPHA = 0.25        # target-point smoothing (lower = smoother, laggier)
DEADBAND_DEG = 1.5      # ignore joint changes smaller than this -> no buzzing
MAX_JOINT_RATE = 60.0   # deg/s slew limit
FACE_TIMEOUT_S = 2.0    # after this long with no face, hold still


# ======================================================================
# 2. MATH HELPERS
# ======================================================================

def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def camera_to_base_rotation(yaw_deg, pitch_deg, roll_deg):
    """Rotation mapping a point in CAMERA coords into BASE coords."""
    # Nominal: cam +x -> base +y, cam +y (down) -> base -z, cam +z (fwd) -> base -x
    M = np.array([[0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    trim = rz(math.radians(yaw_deg)) @ ry(math.radians(-pitch_deg)) @ rx(math.radians(roll_deg))
    return trim @ M


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ======================================================================
# 3. FACE -> 3D POINT IN CAMERA FRAME
# ======================================================================

class FaceLocator:
    """Returns the head-center position in the camera frame, in metres.

    Uses MediaPipe FaceMesh if available (much steadier, better depth), and
    falls back to OpenCV's Haar cascade otherwise.
    """

    def __init__(self):
        self.mode = "haar"
        self.mesh = None
        self.haar = None
        self.t0 = time.time()
        try:
            # mediapipe >= 0.10.30 removed the legacy mp.solutions API; the
            # Tasks API replaces it and needs the .task model file alongside
            # this script.
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._mp = mp
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1)
            self.mesh = vision.FaceLandmarker.create_from_options(opts)
            self.mode = "mesh"
        except Exception as e:
            print("[face] mesh unavailable (%s); falling back to Haar" % e)
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(path)

    def locate(self, bgr):
        h, w = bgr.shape[:2]
        if self.mode == "mesh":
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = self.mesh.detect_for_video(img, int((time.time() - self.t0) * 1000))
            if not res.face_landmarks:
                return None, None
            lm = res.face_landmarks[0]
            # 478-point model: 468-472 = left iris, 473-477 = right iris.
            p_l = np.array([lm[468].x * w, lm[468].y * h])
            p_r = np.array([lm[473].x * w, lm[473].y * h])
            px = 0.5 * (p_l + p_r)
            span_px = float(np.linalg.norm(p_l - p_r))
            if span_px < 2:
                return None, None
            z = FX * IPD_M / span_px
            box = (px[0] - span_px, px[1] - span_px, 2 * span_px, 2 * span_px)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.15, 6, minSize=(70, 70))
            if len(faces) == 0:
                return None, None
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            px = np.array([x + fw / 2.0, y + fh / 2.0])
            z = FX * HEAD_WIDTH_M / float(fw)
            box = (x, y, fw, fh)

        # Pinhole back-projection at the estimated depth.
        p_cam = np.array([(px[0] - CX) * z / FX,
                          (px[1] - CY) * z / FY,
                          z])
        return p_cam, box


# ======================================================================
# 4. POINTING IK
#    J1 swings the arm's vertical plane onto the target.
#    J2+J3 place the wrist center at a fixed comfortable spot in that plane.
#    J4 makes the tool axis look at the target.  J5/J6 stay parked.
# ======================================================================

def solve_wrist_pose():
    """2R IK for the fixed wrist park position. Returns absolute in-plane angles
    (radians, measured from the +rho axis) of the upper arm and the forearm.

    Constant for a given config, so it is solved once at import time.
    """
    dr = WRIST_RHO
    dz = WRIST_Z - D1
    d = math.hypot(dr, dz)
    d = clamp(d, abs(A2 - A3) + 1e-4, A2 + A3 - 1e-4)

    # Elbow-up: rotate the upper arm above the shoulder-wrist line.
    cos_a = clamp((d * d + A2 * A2 - A3 * A3) / (2 * A2 * d), -1.0, 1.0)
    q2_abs = math.atan2(dz, dr) + math.acos(cos_a)

    elbow_rho = A2 * math.cos(q2_abs)
    elbow_z = D1 + A2 * math.sin(q2_abs)
    q3_abs = math.atan2(WRIST_Z - elbow_z, WRIST_RHO - elbow_rho)
    return q2_abs, q3_abs


Q2_ABS, Q3_ABS = solve_wrist_pose()


def point_at(target_base):
    """target_base: np.array([x, y, z]) in BASE frame.

    Returns (dict of joint degrees, reachable_bool).
    """
    tx, ty, tz = target_base

    j1 = math.degrees(math.atan2(ty, tx))      # base yaw toward the target
    rho_t = math.hypot(tx, ty)                 # target radius in the arm plane

    # Elevation the tool must take. Iterate so the TOOL_LEN offset, which moves
    # the tip away from the wrist, is accounted for.
    phi = math.atan2(tz - WRIST_Z, rho_t - WRIST_RHO)
    for _ in range(3):
        tip_rho = WRIST_RHO + TOOL_LEN * math.cos(phi)
        tip_z = WRIST_Z + TOOL_LEN * math.sin(phi)
        dr, dz = rho_t - tip_rho, tz - tip_z
        if math.hypot(dr, dz) < 1e-4:
            break
        phi = math.atan2(dz, dr)

    # Serial-chain angles. Zero pose = arm vertical, so a link's absolute
    # in-plane angle is (90deg + cumulative joint sum). The tool is an extra
    # FLANGE_OFFSET_DEG round from the forearm, which cancels the 90deg and
    # leaves tool elevation == J2 + J3 + J4.
    q2d, q3d, phid = math.degrees(Q2_ABS), math.degrees(Q3_ABS), math.degrees(phi)
    j2 = PITCH_SIGN * (q2d - 90.0)
    j3 = PITCH_SIGN * (q3d - q2d)
    j4 = PITCH_SIGN * (phid - q3d + FLANGE_OFFSET_DEG)

    goal = dict(j1_yaw=j1, j2_shoulder=j2, j3_elbow=j3,
                j4_wrist=j4, j5_wrist2=0.0, j6_roll=0.0)

    reachable = all(j["lo"] <= goal[j["name"]] <= j["hi"] for j in JOINTS)
    return goal, reachable


# ======================================================================
# 5. ROBOT
# ======================================================================

class Arm:
    """Thin wrapper over pymycobot with a dry-run fallback."""

    def __init__(self):
        self.ok = False
        self.mc = None
        self.err = ""
        try:
            from pymycobot.mycobot280 import MyCobot280
            self.mc = MyCobot280(PORT, BAUD)
            time.sleep(0.2)
            # Mode 1 = always execute the newest command; without this the
            # controller queues points and tracking lags badly.
            self.mc.set_fresh_mode(1)
            time.sleep(0.05)
            self.mc.focus_all_servos()
            time.sleep(0.05)
            self.ok = True
        except Exception as e:
            self.err = str(e)
            print("[arm] dry-run mode (%s)" % e)

    def read_angles(self):
        """Current joint angles in degrees, or None if unavailable."""
        if not self.ok:
            return None
        try:
            a = self.mc.get_angles()
        except Exception:
            return None
        if not a or len(a) != 6 or any(v is None for v in a):
            return None
        return [float(v) for v in a]

    def write(self, degrees_by_name):
        if not self.ok:
            return
        angles = [clamp(degrees_by_name[j["name"]], j["lo"], j["hi"]) for j in JOINTS]
        try:
            self.mc.send_angles(angles, MOVE_SPEED)
        except Exception as e:
            print("[arm] send failed: %s" % e)

    def close(self):
        """Leave the arm holding its pose.

        Do NOT call release_all_servos() here - that cuts torque and the arm
        drops onto the desk under its own weight. Torque stays on so it holds
        wherever it stopped.
        """
        if self.ok:
            try:
                self.mc.focus_all_servos()
            except Exception:
                pass


# ======================================================================
# 6. MAIN LOOP
# ======================================================================

def main():
    global CAM_YAW_DEG, CAM_PITCH_DEG

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise SystemExit("could not open camera %d" % CAM_INDEX)

    locator = FaceLocator()
    arm = Arm()

    home = {j["name"]: 0.0 for j in JOINTS}

    # Seed from the real pose so the first command is not a lunge from zero.
    read = arm.read_angles()
    if read is not None:
        current = {j["name"]: read[i] for i, j in enumerate(JOINTS)}
        print("[info] seeded from arm: %s" % ["%.1f" % v for v in read])
    else:
        current = dict(home)
        if arm.ok:
            print("[warn] could not read angles; assuming home pose")

    armed = START_ARMED
    smoothed = None
    last_seen = 0.0
    last_send = 0.0
    reachable = True
    dt = 1.0 / LOOP_HZ
    send_period = 1.0 / SEND_HZ

    print("[info] face locator: %s | arm: %s" % (locator.mode, "live" if arm.ok else "dry-run"))
    print("[info] SPACE to arm motion, q to quit")

    try:
        prev_t = time.time()
        while True:
            t0 = time.time()
            # The mesh detector runs well under LOOP_HZ, so measure the real
            # frame period - a fixed dt would silently scale the slew limit.
            dt_meas = clamp(t0 - prev_t, 1e-3, 0.5)
            prev_t = t0
            ok, frame = cap.read()
            if not ok:
                time.sleep(dt)
                continue

            p_cam, box = locator.locate(frame)
            if p_cam is not None:
                last_seen = t0
                R = camera_to_base_rotation(CAM_YAW_DEG, CAM_PITCH_DEG, CAM_ROLL_DEG)
                p_base = R @ p_cam + CAM_POS_BASE
                smoothed = p_base if smoothed is None else \
                    EMA_ALPHA * p_base + (1 - EMA_ALPHA) * smoothed

            if smoothed is not None and (t0 - last_seen) < FACE_TIMEOUT_S:
                goal, reachable = point_at(smoothed)
                step = MAX_JOINT_RATE * dt_meas
                moved = False
                for j in JOINTS:
                    n = j["name"]
                    err = clamp(goal[n], j["lo"], j["hi"]) - current[n]
                    if abs(err) < DEADBAND_DEG:
                        continue
                    current[n] += clamp(err, -step, step)
                    moved = True
                if moved and armed and (t0 - last_send) >= send_period:
                    arm.write(current)
                    last_send = t0

            # --- overlay
            if box is not None:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            if smoothed is not None:
                cv2.putText(frame, "base xyz  %.2f %.2f %.2f m" % tuple(smoothed),
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "j1 %.0f  j2 %.0f  j3 %.0f  j4 %.0f" %
                            (current["j1_yaw"], current["j2_shoulder"],
                             current["j3_elbow"], current["j4_wrist"]),
                            (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            status = "ARMED" if armed else "DISARMED (space)"
            color = (0, 255, 0) if armed else (0, 165, 255)
            if not reachable:
                status += "  |  OUT OF REACH"
                color = (0, 0, 255)
            if not arm.ok:
                status += "  |  DRY-RUN"
            cv2.putText(frame, status, (12, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "yaw %.1f  pitch %.1f  (a/d, w/s to trim)" %
                        (CAM_YAW_DEG, CAM_PITCH_DEG),
                        (12, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow("face -> mycobot 280", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                armed = not armed
                print("[info] motion %s" % ("ARMED" if armed else "disarmed"))
            elif k == ord("h"):
                current = dict(home)
                if armed:
                    arm.write(current)
            elif k == ord("t"):
                # Known pose: tool axis horizontal, pointing straight forward.
                # If the arm instead folds backward, set PITCH_SIGN = -1.
                test, _ = point_at(np.array([1.0, 0.0, WRIST_Z]))
                current = dict(test)
                if armed:
                    arm.write(current)
                print("[test] %s" % {k2: round(v, 1) for k2, v in test.items()})
            elif k == ord("a"):
                CAM_YAW_DEG -= 1.0
            elif k == ord("d"):
                CAM_YAW_DEG += 1.0
            elif k == ord("w"):
                CAM_PITCH_DEG += 1.0
            elif k == ord("s"):
                CAM_PITCH_DEG -= 1.0
            elif k == ord("p"):
                print("CAM_YAW_DEG = %.1f ; CAM_PITCH_DEG = %.1f" % (CAM_YAW_DEG, CAM_PITCH_DEG))

            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n[info] interrupted - arm holds its current pose")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.close()


if __name__ == "__main__":
    main()
