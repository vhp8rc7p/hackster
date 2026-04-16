import cv2
import cv2.aruco as aruco
import numpy as np
import time
import argparse
from pymycobot import MyPalletizer260
from pump import set_pump

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Find, pick, and place a specific ArUco marker.')
parser.add_argument('--id', type=int, required=True, help='The ArUco ID to pick up.')
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
PICK_Z = 60
DROP_COORDS = [0, -150, 100]
MOVE_SPEED = 40
TOOL_OFFSET_X = -10
TOOL_OFFSET_Y = -40
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
def find_target_marker(cap, target_id, num_frames=10):
    """Looks for a specific ArUco marker and returns its stable position."""
    positions = []
    print(f"正在定位 ID: {target_id}...")
    for _ in range(num_frames * 3):
        ret, frame = cap.read()
        if not ret: continue

        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        if ids is not None and target_id in ids:
            idx = np.where(ids == target_id)[0][0]
            c = corners[idx][0]
            u, v = np.mean(c[:, 0]), np.mean(c[:, 1])
            positions.append((u, v))

        if len(positions) >= num_frames:
            break
        time.sleep(0.1)

    if len(positions) < num_frames:
        print(f"无法稳定定位 ID: {target_id}。")
        return None

    avg_u = int(np.mean([p[0] for p in positions]))
    avg_v = int(np.mean([p[1] for p in positions]))
    print(f"锁定 ID {target_id} 稳定位置: ({avg_u}, {avg_v})")
    return (avg_u, avg_v)

# --- 5. 主执行函数 ---
def main():
    # 1. 移动到初始姿态
    print("移动到初始观察位置...")
    arm.send_angles([0, 0, 0, 0], MOVE_SPEED)
    time.sleep(2.5) # Wait for physical move to complete
    set_pump(arm, 3, False)
    time.sleep(1)

    # Get coordinates at the stable home position, with retries
    home_pos = None
    for i in range(5): # Try up to 5 times
        home_pos = arm.get_coords()
        if isinstance(home_pos, list) and len(home_pos) >= 4:
            print(f"成功获取坐标 on attempt {i+1}: {home_pos}")
            break
        print(f"获取坐标失败 on attempt {i+1} (Result: {home_pos}), retrying...")
        time.sleep(0.5)

    if not (isinstance(home_pos, list) and len(home_pos) >= 4):
        print("多次尝试后仍无法获取机器人当前坐标，任务中止。")
        return

    # 2. 定位目标ID
    marker_pixel_coords = find_target_marker(cap, TARGET_ID)
    if not marker_pixel_coords:
        print("任务中止。")
        return

    # 3. 计算机器人目标坐标
    u, v = marker_pixel_coords
    du, dv = u - u_center, v - v_center
    delta_mm = R_pixel_to_mm.dot(np.array([du, dv]))
    dx_mm = delta_mm[0] * DIR_X
    dy_mm = delta_mm[1] * DIR_Y

    # Use the stored home_pos for calculation
    final_x = home_pos[0] + dx_mm + TOOL_OFFSET_X
    final_y = home_pos[1] + dy_mm + TOOL_OFFSET_Y

    # 4. 执行抓取和放置
    print(f"锁定目标 ID {TARGET_ID}，开始抓放流程...")
    print(f"移动至目标上方: X:{final_x:.1f} Y:{final_y:.1f} Z:{SAFE_Z}")
    arm.send_coords([final_x, final_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(2)
    print(f"下降至抓取高度: Z:{PICK_Z}")
    arm.send_coords([final_x, final_y, PICK_Z, 0], MOVE_SPEED)
    time.sleep(2)
    print("打开气泵，执行抓取")
    set_pump(arm, 3, True)
    time.sleep(1.5)
    print("提升目标")
    arm.send_coords([final_x, final_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(2)
    print(f"移动至放置点: {DROP_COORDS}")
    arm.send_coords(DROP_COORDS + [0], MOVE_SPEED)
    time.sleep(3)
    print("关闭气泵，放置目标")
    set_pump(arm, 3, False)
    time.sleep(2)

    # 5. 回到初始位置
    print("任务完成，回到初始位置。")
    arm.send_angles([0,0,0,0], MOVE_SPEED)
    time.sleep(2)

if __name__ == '__main__':
    main()
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    set_pump(arm, 3, False)
    print("程序已退出。")
