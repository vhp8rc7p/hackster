"""
Hand-follow with proper IK: detect hand → solve IK with continuity from
current joint state → send_angles (no firmware IK branch flipping).

Usage:
  python hand_follow_ik.py
"""

import json
import time
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from pymycobot.mycobot280 import MyCobot280
from ikpy.chain import Chain

# ── config ─────────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.usbserial-54780106801"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
CALIB_PATH = "calibration_result.json"
URDF_PATH = "mycobot_280_m5.urdf"

OWL_QUERIES = ["a hand", "a human hand", "open palm"]
OWL_THRESHOLD = 0.10

# Filter partial-hand detections (when the arm is on top of the hand):
# only accept a bbox if it's at least this fraction of the recent max area.
HAND_AREA_KEEP_FRACTION = 0.70
HAND_AREA_DECAY = 0.995    # recent max decays slowly so we don't get stuck

HAND_Z_BASE_MM = 30.0     # assumed hand height in robot base frame
HOVER_HEIGHT = 150.0
SPEED = 30
MOVE_THRESHOLD_MM = 10
MIN_MOVE_INTERVAL = 0.6
IK_ERR_LIMIT_MM = 20      # if IK can't get within this, target is unreachable
MAX_JOINT_STEP_DEG = 120  # allow big first move, catches truly wild IK flips


# ── IK setup ──────────────────────────────────────────────────────────

# active_links_mask: True for the 6 revolute joints, False for fixed links
chain = Chain.from_urdf_file(
    URDF_PATH,
    base_elements=['g_base'],
    last_link_vector=[0, 0, 0],
    active_links_mask=[False, False, True, True, True, True, True, True, False],
)


def solve_ik(target_xyz_mm, current_angles_deg, pointing_down=True):
    """Return target joint angles (deg) for given TCP target.
    If pointing_down, constrain tool Z axis to base -Z (pump nozzle vertical).
    Uses current angles as IK seed → closest-to-current solution."""
    target_T = np.eye(4)
    target_T[:3, 3] = np.array(target_xyz_mm) / 1000.0

    init = [0.0] * len(chain.links)
    for i, deg in enumerate(current_angles_deg):
        init[i + 2] = float(np.radians(deg))

    if pointing_down:
        # Encode the desired tool orientation directly in target_T.
        # 180° rotation about X axis → tool Z points to base -Z (down).
        target_T[:3, :3] = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ])
        joints = chain.inverse_kinematics_frame(
            target_T, initial_position=init, orientation_mode="Z",
        )
    else:
        joints = chain.inverse_kinematics_frame(
            target_T, initial_position=init, orientation_mode=None,
        )
    return [float(np.degrees(joints[i + 2])) for i in range(6)]


def fk_position_mm(angles_deg):
    """Forward kinematics: joint angles (deg) → TCP position (mm)."""
    pose = [0.0] * len(chain.links)
    for i, deg in enumerate(angles_deg):
        pose[i + 2] = float(np.radians(deg))
    T = chain.forward_kinematics(pose)
    return T[:3, 3] * 1000.0


# ── helpers ───────────────────────────────────────────────────────────

def pixel_to_base_at_z(u, v, mtx, dist, T_cam2base, z_target):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    norm = cv2.undistortPoints(pts, mtx, dist).reshape(2)
    dir_cam = np.array([norm[0], norm[1], 1.0])
    dir_cam /= np.linalg.norm(dir_cam)
    origin_base = T_cam2base[:3, 3]
    dir_base = T_cam2base[:3, :3] @ dir_cam
    if abs(dir_base[2]) < 1e-6: return None
    t = (z_target - origin_base[2]) / dir_base[2]
    if t < 0: return None
    return origin_base + t * dir_base


# ── main ───────────────────────────────────────────────────────────────

def main():
    with open(CALIB_PATH) as f:
        cal = json.load(f)
    mtx = np.array(cal["camera_matrix"])
    dist = np.array(cal["dist_coeffs"])
    T_cam2base = np.array(cal["T_cam2base"])
    print(f"Calibration loaded. cam_in_base={T_cam2base[:3,3].round(1)}")

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
    print("Ready. Show hand. 'q' in preview to quit.\n")

    last_target = None
    last_move_time = 0.0
    recent_max_area = 0.0   # for filtering partial detections

    try:
        while True:
            ret, frame = cap.read()
            if not ret: time.sleep(0.05); continue

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
                cv2.imshow("hand_follow_ik", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            x1, y1, x2, y2 = [int(v) for v in best[0]]
            cx_px, cy_px = (x1 + x2) // 2, (y1 + y2) // 2
            area = max(0, x2 - x1) * max(0, y2 - y1)

            # Track recent max area, decay so we don't lock in forever.
            recent_max_area = max(area, recent_max_area * HAND_AREA_DECAY)
            # Reject if bbox is much smaller than recent max — probably partial.
            if area < recent_max_area * HAND_AREA_KEEP_FRACTION:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange
                cv2.putText(disp, f"partial (area {area:.0f}/{recent_max_area:.0f})", (x1, max(y1-8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.imshow("hand_follow_ik", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(disp, (cx_px, cy_px), 8, (0, 255, 255), -1)

            hand_base = pixel_to_base_at_z(cx_px, cy_px, mtx, dist, T_cam2base, HAND_Z_BASE_MM)
            if hand_base is None:
                cv2.imshow("hand_follow_ik", disp)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                continue

            tcp_target_xyz = hand_base + np.array([0, 0, HOVER_HEIGHT])

            txt = f"hand x={hand_base[0]:.0f} y={hand_base[1]:.0f} z={hand_base[2]:.0f}"
            cv2.putText(disp, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("hand_follow_ik", disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break

            if tcp_target_xyz[2] > 400 or tcp_target_xyz[2] < 30:
                print(f"  z out of range {tcp_target_xyz[2]:.0f}"); continue

            now = time.time()
            moved = (last_target is None
                     or np.linalg.norm(tcp_target_xyz - last_target) > MOVE_THRESHOLD_MM)
            cooled = (now - last_move_time) > MIN_MOVE_INTERVAL
            if not (moved and cooled):
                continue

            current_angles = mc.get_angles()
            if not current_angles or len(current_angles) != 6:
                print("  no angles, skipping"); continue

            # Try strict-down first; fall back to position-only if it can't reach.
            mode_used = "down"
            try:
                target_angles = solve_ik(tcp_target_xyz, current_angles, pointing_down=True)
            except Exception as e:
                print(f"  IK solve failed: {e}"); continue
            fk_pos = fk_position_mm(target_angles)
            ik_err = float(np.linalg.norm(fk_pos - tcp_target_xyz))
            if ik_err > IK_ERR_LIMIT_MM:
                try:
                    target_angles = solve_ik(tcp_target_xyz, current_angles, pointing_down=False)
                except Exception as e:
                    print(f"  IK fallback failed: {e}"); continue
                fk_pos = fk_position_mm(target_angles)
                ik_err = float(np.linalg.norm(fk_pos - tcp_target_xyz))
                mode_used = "tilted"
                if ik_err > IK_ERR_LIMIT_MM:
                    print(f"  ⚠ unreachable even with tilt (err={ik_err:.0f}mm)")
                    continue

            joint_change = max(abs(a - b) for a, b in zip(target_angles, current_angles))
            if joint_change > MAX_JOINT_STEP_DEG:
                print(f"  ⚠ joint swing too large ({joint_change:.0f}°) — skipping")
                continue

            try:
                mc.send_angles(target_angles, SPEED)
            except Exception as e:
                print(f"  send_angles failed: {e}"); continue

            time.sleep(0.8)
            print(f"  hand@{hand_base.round(0)}  TCP@{tcp_target_xyz.round(0)}  "
                  f"mode={mode_used}  err={ik_err:.0f}mm  Δjoint={joint_change:.1f}°")
            last_target = tcp_target_xyz.copy()
            last_move_time = now

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)


if __name__ == "__main__":
    main()
