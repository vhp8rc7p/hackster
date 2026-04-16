import cv2
import cv2.aruco as aruco
import numpy as np
import time
import argparse
from pymycobot import MyPalletizer260
from pump import set_pump

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Pick a red cube and place it on a specific ArUco marker.')
parser.add_argument('--id', type=int, required=True, help='The target ArUco marker ID for placing the cube.')
args = parser.parse_args()

# --- 1. 初始化 ---
try:
    arm = MyPalletizer260("COM4", 115200)
    print("机器人连接成功。")
except Exception as e:
    print(f"机器人连接失败: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开相机")
    exit()
h, w, _ = cap.read()[1].shape
u_center, v_center = w / 2, h / 2
print("相机初始化成功。")

# --- 2. 参数 ---
TARGET_ID = args.id
DIR_X = -1
DIR_Y = -1
SAFE_Z = 200
PICK_Z = 75      # The height for picking up the cube
PLACE_Z = 85     # NEW: The height for placing the cube (slightly higher to avoid collision)
MOVE_SPEED = 40
TOOL_OFFSET_X = -10
TOOL_OFFSET_Y = -40
LOWER_RED_1 = np.array([0, 120, 70])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 120, 70])
UPPER_RED_2 = np.array([180, 255, 255])
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# --- 3. 加载标定矩阵 ---
try:
    M = np.load("calibration_matrix.npy")
    R_pixel_to_mm = M[:, :2]
    print("成功加载标定矩阵。")
except FileNotFoundError:
    print("错误：找不到 calibration_matrix.npy。")
    exit()

# --- 4. 功能函数 ---
def find_stable_position(cap, find_func, num_frames=10, **kwargs):
    """Generic function to find a stable position for either a cube or a marker."""
    positions = []
    print(f"正在定位目标: {find_func.__name__}...")
    for _ in range(num_frames * 3):
        ret, frame = cap.read()
        if not ret: continue
        pos = find_func(frame, **kwargs)
        if pos:
            positions.append(pos)
        if len(positions) >= num_frames:
            break
        time.sleep(0.1)
    if len(positions) < num_frames:
        return None
    avg_u = int(np.mean([p[0] for p in positions]))
    avg_v = int(np.mean([p[1] for p in positions]))
    return (avg_u, avg_v)

def find_red_cube(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1) + cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def find_target_marker(frame, target_id):
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None and target_id in ids:
        idx = np.where(ids == target_id)[0][0]
        c = corners[idx][0]
        return (np.mean(c[:, 0]), np.mean(c[:, 1]))
    return None

def calculate_world_coords(pixel_coords, home_pos):
    u, v = pixel_coords
    du, dv = u - u_center, v - v_center
    delta_mm = R_pixel_to_mm.dot(np.array([du, dv]))
    dx_mm = delta_mm[0] * DIR_X
    dy_mm = delta_mm[1] * DIR_Y
    final_x = home_pos[0] + dx_mm + TOOL_OFFSET_X
    final_y = home_pos[1] + dy_mm + TOOL_OFFSET_Y
    return final_x, final_y

# --- 5. 主执行函数 ---
def main():
    print("移动到初始观察位置...")
    arm.send_angles([0, 0, 0, 0], MOVE_SPEED)
    time.sleep(2.5)
    set_pump(arm, 3, False)
    time.sleep(1)

    home_pos = None
    for i in range(5):
        home_pos = arm.get_coords()
        if isinstance(home_pos, list): break
        time.sleep(0.5)
    if not isinstance(home_pos, list):
        print("无法获取机器人坐标，任务中止。")
        return

    # Step 1: Find the cube
    cube_pixels = find_stable_position(cap, find_red_cube)
    if not cube_pixels:
        print("找不到红色木块，任务中止。")
        return
    print(f"锁定木块位置: {cube_pixels}")

    # Step 2: Find the marker
    marker_pixels = find_stable_position(cap, find_target_marker, target_id=TARGET_ID)
    if not marker_pixels:
        print(f"找不到 ArUco marker ID {TARGET_ID}，任务中止。")
        return
    print(f"锁定 Marker ID {TARGET_ID} 位置: {marker_pixels}")

    # Step 3: Calculate both world coordinates
    pick_x, pick_y = calculate_world_coords(cube_pixels, home_pos)
    place_x, place_y = calculate_world_coords(marker_pixels, home_pos)

    print(f"计算拾取坐标: X:{pick_x:.1f}, Y:{pick_y:.1f}")
    print(f"计算放置坐标: X:{place_x:.1f}, Y:{place_y:.1f}")

    # Step 4: Execute the sequence
    print("\n开始执行'pick and place'流程...")
    # Pick up the cube
    arm.send_coords([pick_x, pick_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(2)
    arm.send_coords([pick_x, pick_y, PICK_Z, 0], MOVE_SPEED)
    time.sleep(2)
    set_pump(arm, 3, True)
    time.sleep(1.5)
    arm.send_coords([pick_x, pick_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(2)

    # Place on the marker
    arm.send_coords([place_x, place_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(3)
    arm.send_coords([place_x, place_y, PLACE_Z, 0], MOVE_SPEED)
    time.sleep(2)
    set_pump(arm, 3, False)
    time.sleep(1)
    arm.send_coords([place_x, place_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(2)

    # Return home
    print("任务完成，回到初始位置。")
    arm.send_angles([0,0,0,0], MOVE_SPEED)
    time.sleep(2)

if __name__ == '__main__':
    main()
    cap.release()
    cv2.destroyAllWindows()
    set_pump(arm, 3, False)
    print("程序已退出。")
