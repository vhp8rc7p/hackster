import time
import argparse
import numpy as np
from pymycobot import MyPalletizer260
from pump import set_pump

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Pick an object at a specific pixel coordinate.')
parser.add_argument('--u', type=int, required=True, help='The U (horizontal) pixel coordinate of the target.')
parser.add_argument('--v', type=int, required=True, help='The V (vertical) pixel coordinate of the target.')
args = parser.parse_args()

# --- 1. Parameters ---
PICK_PIXELS = (args.u, args.v)
# Robot Parameters
DIR_X = -1
DIR_Y = -1
SAFE_Z = 200
PICK_Z = 75
DROP_COORDS = [0, -150, 100]
MOVE_SPEED = 40
TOOL_OFFSET_X = -10
TOOL_OFFSET_Y = -40
H, W = 480, 640 # Assuming standard camera dimensions
U_CENTER, V_CENTER = W / 2, H / 2

# --- 2. Initialization ---
try:
    arm = MyPalletizer260('COM4', 115200)
    print('机器人连接成功。')
except Exception as e:
    print(f'机器人连接失败: {e}')
    exit()

try:
    M = np.load('calibration_matrix.npy')
    R_pixel_to_mm = M[:, :2]
    print('成功加载标定矩阵。')
except FileNotFoundError:
    print('错误：找不到 calibration_matrix.npy。')
    exit()

# --- 3. Main Execution ---
def main():
    print('移动到初始观察位置...')
    arm.send_angles([0, 0, 0, 0], MOVE_SPEED)
    time.sleep(2.5)
    set_pump(arm, 3, False)
    time.sleep(1)

    # Get home coordinates with retry
    home_pos = None
    for i in range(5):
        home_pos = arm.get_coords()
        if isinstance(home_pos, list) and len(home_pos) >= 4:
            print(f'成功获取坐标 on attempt {i+1}: {home_pos}')
            break
        print(f'获取坐标失败 on attempt {i+1}, retrying...')
        time.sleep(0.5)

    if not isinstance(home_pos, list):
        print('多次尝试后仍无法获取机器人当前坐标，任务中止。')
        exit()

    # Calculate world coordinates from the provided pixel arguments
    u, v = PICK_PIXELS
    du, dv = u - U_CENTER, v - V_CENTER
    delta_mm = R_pixel_to_mm.dot(np.array([du, dv]))
    dx_mm = delta_mm[0] * DIR_X
    dy_mm = delta_mm[1] * DIR_Y
    final_x = home_pos[0] + dx_mm + TOOL_OFFSET_X
    final_y = home_pos[1] + dy_mm + TOOL_OFFSET_Y

    print(f'根据像素 ({u},{v}) 计算拾取坐标: X:{final_x:.1f}, Y:{final_y:.1f}')

    # Execute pick and place
    print('\n开始执行 pick and place 流程...')
    arm.send_coords([final_x, final_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(2)
    arm.send_coords([final_x, final_y, PICK_Z, 0], MOVE_SPEED)
    time.sleep(2)
    set_pump(arm, 3, True)
    time.sleep(1.5)
    arm.send_coords([final_x, final_y, SAFE_Z, 0], MOVE_SPEED)
    time.sleep(2)
    arm.send_coords(DROP_COORDS + [0], MOVE_SPEED)
    time.sleep(3)
    set_pump(arm, 3, False)
    time.sleep(1)

    # Return home
    print('任务完成，回到初始位置。')
    arm.send_angles([0,0,0,0], MOVE_SPEED)
    time.sleep(2)

    set_pump(arm, 3, False)
    print('程序已退出。')

if __name__ == '__main__':
    main()
