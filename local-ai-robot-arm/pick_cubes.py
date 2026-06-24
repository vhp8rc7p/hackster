"""
Detect purple and yellow cubes with OWL-ViT, position the suction pump
tip on top of each. Uses IK from the URDF, send_angles for execution.

No suction is actuated — just touches the top of each cube.

Usage:
  python pick_cubes.py
"""

import json
import time
import sys
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

# Cube geometry — set this to your cubes' actual size in mm.
CUBE_HEIGHT_MM = 25.0
# Table surface z in robot base frame. Touch calibration assumed table=0,
# but adjust if your marker rested on something taller.
TABLE_Z_BASE_MM = 0.0

# Move heights (mm above cube TOP):
HOVER_ABOVE_CUBE = 80.0
TOUCH_ABOVE_CUBE = 5.0     # how close the pump tip gets to the cube top

PUMP_LENGTH = 50.0
SPEED = 30
IK_ERR_LIMIT_MM = 15
MAX_JOINT_STEP_DEG = 150   # generous since this is a one-shot script

OWL_QUERIES = ["a small pink object", "a small yellow object"]
OWL_THRESHOLD = 0.15

HOME_ANGLES = [0, 0, 0, 0, 0, 0]


# ── IK setup ──────────────────────────────────────────────────────────
chain = Chain.from_urdf_file(
    URDF_PATH, base_elements=['g_base'], last_link_vector=[0, 0, 0],
    active_links_mask=[False, False, True, True, True, True, True, True, False],
)


def solve_ik(target_xyz_mm, current_angles_deg):
    """Pump-down IK: tool Z axis locked to base -Z."""
    target_T = np.eye(4)
    target_T[:3, 3] = np.array(target_xyz_mm) / 1000.0
    target_T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    init = [0.0] * len(chain.links)
    for i, deg in enumerate(current_angles_deg):
        init[i + 2] = float(np.radians(deg))
    joints = chain.inverse_kinematics_frame(
        target_T, initial_position=init, orientation_mode="Z",
    )
    return [float(np.degrees(joints[i + 2])) for i in range(6)]


def fk_pos_mm(angles_deg):
    pose = [0.0] * len(chain.links)
    for i, deg in enumerate(angles_deg):
        pose[i + 2] = float(np.radians(deg))
    return chain.forward_kinematics(pose)[:3, 3] * 1000.0


# ── geometry helpers ─────────────────────────────────────────────────

def pixel_to_base_at_z(u, v, mtx, dist, T_cam2base, z_target):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    norm = cv2.undistortPoints(pts, mtx, dist).reshape(2)
    d = np.array([norm[0], norm[1], 1.0]); d /= np.linalg.norm(d)
    origin = T_cam2base[:3, 3]
    dir_b = T_cam2base[:3, :3] @ d
    if abs(dir_b[2]) < 1e-6: return None
    t = (z_target - origin[2]) / dir_b[2]
    if t < 0: return None
    return origin + t * dir_b


def tcp_target_for_pump_tip(pump_tip_xyz_mm):
    """When pump points straight down, TCP is PUMP_LENGTH above the tip."""
    return np.array([pump_tip_xyz_mm[0], pump_tip_xyz_mm[1],
                     pump_tip_xyz_mm[2] + PUMP_LENGTH])


# ── motion ────────────────────────────────────────────────────────────

def move_to_tcp(mc, target_xyz_mm, label=""):
    """Solve IK from current pose, verify convergence, send angles."""
    current = mc.get_angles()
    if not current or len(current) != 6:
        print(f"  [{label}] no angles, abort"); return False
    try:
        target_angles = solve_ik(target_xyz_mm, current)
    except Exception as e:
        print(f"  [{label}] IK exception: {e}"); return False
    fk = fk_pos_mm(target_angles)
    err = float(np.linalg.norm(fk - target_xyz_mm))
    if err > IK_ERR_LIMIT_MM:
        print(f"  [{label}] IK didn't converge (err={err:.0f}mm)  TCP target={target_xyz_mm.round(1)}")
        return False
    swing = max(abs(a - b) for a, b in zip(target_angles, current))
    if swing > MAX_JOINT_STEP_DEG:
        print(f"  [{label}] joint swing too large ({swing:.0f}°)"); return False
    try:
        mc.send_angles(target_angles, SPEED)
    except Exception as e:
        print(f"  [{label}] send_angles failed: {e}"); return False
    time.sleep(2.5)
    print(f"  [{label}] TCP target {target_xyz_mm.round(1)}  ik_err={err:.0f}mm  swing={swing:.0f}°")
    return True


def visit_cube(mc, cube_xy_mm, label):
    """Hover above cube, lower pump to just above top, lift back to hover."""
    cube_top_z = TABLE_Z_BASE_MM + CUBE_HEIGHT_MM
    hover_tip   = np.array([cube_xy_mm[0], cube_xy_mm[1], cube_top_z + HOVER_ABOVE_CUBE])
    touch_tip   = np.array([cube_xy_mm[0], cube_xy_mm[1], cube_top_z + TOUCH_ABOVE_CUBE])
    hover_tcp = tcp_target_for_pump_tip(hover_tip)
    touch_tcp = tcp_target_for_pump_tip(touch_tip)

    print(f"\n→ {label} at base xy=({cube_xy_mm[0]:.0f}, {cube_xy_mm[1]:.0f})")
    if not move_to_tcp(mc, hover_tcp, f"{label} hover"): return
    if not move_to_tcp(mc, touch_tcp, f"{label} touch"): return
    time.sleep(0.7)
    move_to_tcp(mc, hover_tcp, f"{label} lift")


# ── main ──────────────────────────────────────────────────────────────

def main():
    with open(CALIB_PATH) as f:
        cal = json.load(f)
    mtx = np.array(cal["camera_matrix"])
    dist = np.array(cal["dist_coeffs"])
    T_cam2base = np.array(cal["T_cam2base"])
    print(f"Calibration: cam_in_base={T_cam2base[:3, 3].round(1)}")

    print("Loading OWL-ViT...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    print(f"  device={device}")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # Robot connection delayed — capture frame first so connection stays fresh.
    print("\nCapturing frame...")
    for _ in range(10): cap.read()  # flush
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("camera read failed"); return

    cv2.imwrite("pick_cubes_frame.png", frame)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=[OWL_QUERIES], images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.Tensor([pil.size[::-1]])
    res = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=OWL_THRESHOLD)[0]

    # pick best detection per query label
    best_per_query = {q: None for q in OWL_QUERIES}
    for box, score, label_idx in zip(res["boxes"], res["scores"], res["labels"]):
        q = OWL_QUERIES[int(label_idx)]
        s = float(score)
        if best_per_query[q] is None or s > best_per_query[q][1]:
            best_per_query[q] = (box.tolist(), s)

    # back-project each cube center; cube TOP is at z = TABLE + CUBE_HEIGHT
    cube_top_z = TABLE_Z_BASE_MM + CUBE_HEIGHT_MM
    cube_positions = []
    print("\nDetections:")
    annotated = frame.copy()
    for q, det in best_per_query.items():
        if det is None:
            print(f"  {q:13s}: NOT FOUND"); continue
        box, score = det
        x1, y1, x2, y2 = [int(v) for v in box]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Back-project to z = cube top (the top face is what the camera sees)
        base = pixel_to_base_at_z(cx, cy, mtx, dist, T_cam2base, cube_top_z)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{q} {score:.2f}", (x1, max(y1-8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if base is None:
            print(f"  {q:13s}: detected but back-projection failed"); continue
        print(f"  {q:13s}: conf={score:.2f}  base=({base[0]:.0f}, {base[1]:.0f}, {base[2]:.0f})")
        cube_positions.append((q, base[:2]))
    cv2.imwrite("pick_cubes_annotated.png", annotated)
    print("Annotated frame saved → pick_cubes_annotated.png")

    if not cube_positions:
        print("\nNo cubes found — adjust queries / threshold / lighting"); return

    print("\nConnecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    mc.power_on()
    time.sleep(0.5)

    print("Homing to zero...")
    mc.send_angles(HOME_ANGLES, SPEED)
    time.sleep(4)

    for label, xy in cube_positions:
        visit_cube(mc, xy, label)

    print("\nReturning home...")
    mc.send_angles(HOME_ANGLES, SPEED)
    time.sleep(4)
    print("Done.")


if __name__ == "__main__":
    main()
