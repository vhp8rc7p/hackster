"""
OWL-ViT based hand-follow: detect hand in gantry camera, move arm so the
suction pump hovers above it. No ArUco needed.

Uses T_cam2base from calibration_result.json (touch calibration).
Hand 3D position is estimated by intersecting the camera ray through the
bbox center with an assumed Z plane (set HAND_Z_BASE_MM below).

Usage:
  python hand_follow.py
"""

import json
import time
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from pymycobot.mycobot280 import MyCobot280

# ── config ─────────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080

CALIB_PATH = "calibration_result.json"

OWL_QUERIES = ["a hand", "a human hand", "open palm"]
OWL_THRESHOLD = 0.10

# We can't get true depth from a single image — assume the hand is at
# this Z height in robot BASE frame (mm). 0 = desk surface, 80 = hand
# resting palm-down on desk, etc. Adjust to match where your hand actually is.
HAND_Z_BASE_MM = 30.0

PUMP_LENGTH = 50.0
HOVER_HEIGHT = 150.0
SPEED = 30

TOOL_RX, TOOL_RY, TOOL_RZ = -160.0, 0.0, 0.0  # tilted ~20° to avoid singularity

MOVE_THRESHOLD_MM = 10
MIN_MOVE_INTERVAL = 0.5     # can be tighter — increments are small
REACH_LIMIT_MM = 300
MAX_STEP_MM = 20            # clamp per-axis jog to this; smaller = smoother


# ── helpers ────────────────────────────────────────────────────────────

def tool_rotation(rxd, ryd, rzd):
    rx, ry, rz = np.radians([rxd, ryd, rzd])
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx


def tcp_for_pump_tip(pump_tip_xyz):
    R = tool_rotation(TOOL_RX, TOOL_RY, TOOL_RZ)
    offset_base = R @ np.array([0, 0, PUMP_LENGTH])
    tcp_xyz = np.array(pump_tip_xyz) - offset_base
    return [float(tcp_xyz[0]), float(tcp_xyz[1]), float(tcp_xyz[2]),
            TOOL_RX, TOOL_RY, TOOL_RZ]


def pixel_to_base_at_z(u, v, mtx, dist, T_cam2base, z_base_target):
    """Back-project pixel (u,v) to a 3D point in base frame whose Z = z_base_target."""
    # undistort the pixel back to a normalized ray direction in cam frame
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    norm = cv2.undistortPoints(pts, mtx, dist).reshape(2)
    dir_cam = np.array([norm[0], norm[1], 1.0])
    dir_cam /= np.linalg.norm(dir_cam)

    # ray in base frame: origin = T_cam2base translation, dir = T_cam2base rot @ dir_cam
    origin_base = T_cam2base[:3, 3]
    dir_base = T_cam2base[:3, :3] @ dir_cam

    if abs(dir_base[2]) < 1e-6:
        return None  # ray parallel to target plane
    t = (z_base_target - origin_base[2]) / dir_base[2]
    if t < 0:
        return None  # plane behind camera
    return origin_base + t * dir_base


# ── main ───────────────────────────────────────────────────────────────

def main():
    with open(CALIB_PATH) as f:
        cal = json.load(f)
    mtx = np.array(cal["camera_matrix"])
    dist = np.array(cal["dist_coeffs"])
    T_cam2base = np.array(cal["T_cam2base"])
    print(f"Calibration loaded.  cam_in_base={T_cam2base[:3,3].round(1)}")

    print("Loading OWL-ViT...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    print(f"  device={device}")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    print("Connecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    mc.power_on()
    time.sleep(0.5)
    print("Ready. Show your hand in view. Press 'q' in preview to quit.\n")

    last_target = None
    last_move_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05); continue

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(text=[OWL_QUERIES], images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.Tensor([pil.size[::-1]])
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=OWL_THRESHOLD)[0]

            best = None
            for box, score, _ in zip(results["boxes"], results["scores"], results["labels"]):
                if best is None or score > best[1]:
                    best = (box.tolist(), float(score))

            disp = frame.copy()
            if best is None:
                cv2.putText(disp, "no hand", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("hand_follow", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            x1, y1, x2, y2 = [int(v) for v in best[0]]
            cx_px, cy_px = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(disp, (cx_px, cy_px), 8, (0, 255, 255), -1)
            cv2.putText(disp, f"hand {best[1]:.2f}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            hand_base = pixel_to_base_at_z(cx_px, cy_px, mtx, dist, T_cam2base, HAND_Z_BASE_MM)
            if hand_base is None:
                cv2.imshow("hand_follow", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            txt = f"base: x={hand_base[0]:.0f} y={hand_base[1]:.0f} z={hand_base[2]:.0f}"
            cv2.putText(disp, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            pump_tip_target = hand_base + np.array([0, 0, HOVER_HEIGHT])
            tcp_cmd = tcp_for_pump_tip(pump_tip_target)
            reach = float(np.linalg.norm(tcp_cmd[:3]))

            cv2.imshow("hand_follow", disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break

            if reach > REACH_LIMIT_MM:
                print(f"  out of reach |TCP|={reach:.0f}mm (hand@{hand_base.round(0)})")
                continue
            if tcp_cmd[2] > 400 or tcp_cmd[2] < 30:
                print(f"  TCP z out of range: {tcp_cmd[2]:.0f}")
                continue

            target_xyz = np.array(tcp_cmd[:3])
            now = time.time()
            moved = (last_target is None
                     or np.linalg.norm(target_xyz - last_target) > MOVE_THRESHOLD_MM)
            cooled = (now - last_move_time) > MIN_MOVE_INTERVAL

            if moved and cooled:
                # Back to send_coords — it can find IK solutions that per-axis
                # send_coord can't. Use sync version so we don't queue up commands
                # while the arm is still moving.
                current = mc.get_coords()
                if not current or len(current) != 6:
                    print("  no TCP read, skipping"); continue
                # Step toward goal so big hand jumps don't cause big IK flips
                step_vec = np.array(tcp_cmd[:3]) - np.array(current[:3])
                step_mag = np.linalg.norm(step_vec)
                if step_mag > MAX_STEP_MM:
                    step_vec = step_vec * (MAX_STEP_MM / step_mag)
                sub_target = list(np.array(current[:3]) + step_vec) + tcp_cmd[3:]
                try:
                    mc.sync_send_coords(sub_target, SPEED, 0, timeout=4)
                except Exception as e:
                    print(f"  sync_send_coords failed: {e}"); continue
                actual = mc.get_coords()
                err = np.linalg.norm(np.array(actual[:3]) - np.array(sub_target[:3])) if actual else 999
                print(f"  hand@{hand_base.round(0)}  target{[round(v,0) for v in sub_target[:3]]}  err={err:.0f}mm")
                last_target = target_xyz.copy()
                last_move_time = now

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)


if __name__ == "__main__":
    main()
