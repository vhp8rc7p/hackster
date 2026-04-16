import cv2
import cv2.aruco as aruco
import numpy as np
import time
from pymycobot import MyPalletizer260
import time
arm=MyPalletizer260("COM4",115200)
arm.send_coords([165,0,200,0],20)
time.sleep(1)
arm.send_angle(4,0,30)
# ----------------------------------------------

# 配置 ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
cap = cv2.VideoCapture(0)

# 标定参数设置
step_size = 30  # 每次移动 30mm，根据你的相机视野调整
image_points = []
robot_points = []

def capture_point():
    """在当前位置捕获图像坐标和机械臂坐标"""
    # 等待机械臂稳定
    time.sleep(1.5) 
    
    # 读取多次图像取平均值增加精度
    temp_u, temp_v = [], []
    for _ in range(10):
        ret, frame = cap.read()
        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        if ids is not None:
            c = corners[0][0]
            temp_u.append(np.mean(c[:, 0]))
            temp_v.append(np.mean(c[:, 1]))
    
    if len(temp_u) > 0:
        current_robot_pos = arm.get_coords()
        print("z:",current_robot_pos[2])
        return (np.mean(temp_u), np.mean(temp_v)), (current_robot_pos[0], current_robot_pos[1])
    return None, None

# 1. 准备阶段
print("请手动将机械臂移至标定板上方，确保 ArUco 码在画面中心，然后按回车开始...")
input()

# 获取起始中心位置
start_pos = arm.get_coords()
x0, y0, z0, rz0 = start_pos[0], start_pos[1], start_pos[2], start_pos[3]

# 2. 自动化执行九宫格移动
# 偏移量列表：(dx, dy)
offsets = [
    (-1, 1),  (0, 1),  (1, 1),
    (-1, 0),  (0, 0),  (1, 0),
    (-1, -1), (0, -1), (1, -1)
]

print("开始自动采样...")
for dx, dy in offsets:
    target_x = x0 + dx * step_size
    target_y = y0 + dy * step_size
    
    print(f"移动至: X={target_x}, Y={target_y}")
    arm.send_coords([target_x, target_y, z0, rz0],30)
    time.sleep(1.5)
    img_pt, rob_pt = capture_point()
    if img_pt:
        image_points.append(img_pt)
        robot_points.append(rob_pt)
        print(f"成功捕获点: 像素{img_pt} -> 机械臂{rob_pt}")
    else:
        print("警告：未检测到标志，跳过此点")

# 3. 计算标定矩阵
if len(image_points) >= 3:
    pts_src = np.array(image_points, dtype=np.float32)
    pts_dst = np.array(robot_points, dtype=np.float32)
    M, _ = cv2.estimateAffine2D(pts_src, pts_dst)
    
    print("\n标定完成！矩阵 M 为：")
    print(M)
    
    # 保存矩阵供以后使用
    np.save("calibration_matrix.npy", M)
else:
    print("采集点位不足，标定失败。")

cap.release()