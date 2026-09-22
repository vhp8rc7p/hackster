"""
Trace hand_follow_ik.py pipeline. Tests each stage independently so we
can see where it's actually failing — without needing a hand in view.
"""
import json, time, sys
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from pymycobot.mycobot280 import MyCobot280
from ikpy.chain import Chain

CALIB_PATH = "calibration_result.json"
URDF_PATH = "mycobot_280_m5.urdf"
CAMERA_ID = 0
SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200


def step(n, msg):
    print(f"\n=== Step {n}: {msg} ===")


step(1, "Load calibration")
with open(CALIB_PATH) as f:
    cal = json.load(f)
mtx = np.array(cal["camera_matrix"])
dist = np.array(cal["dist_coeffs"])
T_cam2base = np.array(cal["T_cam2base"])
print(f"OK: cam_in_base = {T_cam2base[:3, 3].round(1)}")

step(2, "Load URDF + ikpy chain")
chain = Chain.from_urdf_file(
    URDF_PATH, base_elements=['g_base'], last_link_vector=[0, 0, 0],
    active_links_mask=[False, False, True, True, True, True, True, True, False],
)
print(f"OK: {len(chain.links)} links, 6 active joints")

step(3, "Test FK with a synthetic angle set")
test_angles_deg = [10, -30, 20, 0, 0, 0]
pose = [0.0] * len(chain.links)
for i, deg in enumerate(test_angles_deg): pose[i + 2] = np.radians(deg)
T = chain.forward_kinematics(pose)
print(f"OK: FK([{test_angles_deg}]) → pos(mm)={(T[:3, 3] * 1000).round(1)}")

step(4, "Test IK on a known-reachable target")
fake_hand_base = np.array([-150.0, 50.0, 30.0])
target_tcp = fake_hand_base + np.array([0, 0, 150])  # hover 150mm
print(f"  fake hand@{fake_hand_base}, TCP target = {target_tcp}, |target|={np.linalg.norm(target_tcp):.0f}mm")
target_T = np.eye(4)
target_T[:3, 3] = target_tcp / 1000
init = [0.0] * len(chain.links)
try:
    joints_rad = chain.inverse_kinematics_frame(target_T, initial_position=init, orientation_mode=None)
    joints_deg = [round(np.degrees(joints_rad[i + 2]), 1) for i in range(6)]
    # Verify by FK
    T_check = chain.forward_kinematics(joints_rad)
    pos_check = T_check[:3, 3] * 1000
    err = np.linalg.norm(pos_check - target_tcp)
    print(f"  IK solution (deg): {joints_deg}")
    print(f"  FK back: pos={pos_check.round(1)}  err={err:.1f}mm")
    if err > 50:
        print(f"  ⚠ IK didn't converge close — target may be out of reach")
except Exception as e:
    print(f"  IK FAILED: {e}")
    sys.exit(1)

step(5, "Connect to robot, read state")
mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
time.sleep(2)
mc.power_on()
time.sleep(0.5)
cur_angles = mc.get_angles()
cur_coords = mc.get_coords()
print(f"  angles: {cur_angles}")
print(f"  coords: {cur_coords}")

step(6, "FK on current angles vs robot-reported coords")
pose = [0.0] * len(chain.links)
for i, deg in enumerate(cur_angles): pose[i + 2] = np.radians(deg)
fk_pos = (chain.forward_kinematics(pose)[:3, 3] * 1000)
robot_pos = np.array(cur_coords[:3])
diff = fk_pos - robot_pos
print(f"  URDF FK:  {fk_pos.round(1)}")
print(f"  robot:    {robot_pos.round(1)}")
print(f"  diff:     {diff.round(1)} mm  (|err|={np.linalg.norm(diff):.1f})")
if np.linalg.norm(diff) > 30:
    print(f"  ⚠ URDF frame may be misaligned with robot's reported frame")

step(7, "Load OWL model")
t0 = time.time()
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
print(f"  loaded in {time.time()-t0:.1f}s, device={device}")

step(8, "Capture one frame, run OWL with various thresholds")
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
for _ in range(5): cap.read()
ret, frame = cap.read()
cap.release()
print(f"  frame: {frame.shape if ret else 'FAILED'}")
if ret:
    cv2.imwrite("debug_owl_frame.png", frame)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    queries = ["a hand", "a human hand", "open palm", "fingers", "person", "robot arm"]
    inputs = processor(text=[queries], images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.Tensor([pil.size[::-1]])
    for thresh in [0.05, 0.10, 0.15, 0.20]:
        res = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=thresh)[0]
        labs = [(queries[int(l)], round(float(s), 3)) for s, l in zip(res['scores'], res['labels'])]
        print(f"  threshold={thresh}: {len(res['boxes'])} detections: {labs[:5]}")

print("\nDone. See debug_owl_frame.png for the live camera view.")
