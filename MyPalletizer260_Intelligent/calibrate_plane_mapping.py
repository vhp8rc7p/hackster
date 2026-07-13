"""
平面映射标定 - 用于固定高度的抓取（MyPalletizer 260 版本）
使用棋盘格建立像素坐标到机械臂坐标的映射
"""
import cv2
from uvc_camera import UVCCamera
import numpy as np
import json
from pymycobot import MyPalletizer260
import time

def find_chessboard(frame, pattern_size=(9, 6)):
    """检测棋盘格角点"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)

    return ret, corners, frame

def main():
    print("\n" + "="*70)
    print("平面映射标定 - 固定高度抓取 (MyPalletizer 260)")
    print("="*70)
    print("\n说明：")
    print("1. 将棋盘格平放在A区域（识别区域）")
    print("2. 机械臂移动到固定的观测高度")
    print("3. 程序会检测棋盘格角点")
    print("4. 手动移动机械臂末端到4个角点位置")
    print("5. 建立像素到机械臂坐标的映射关系")
    print("\n" + "="*70 + "\n")

    # 初始化
    mc = MyPalletizer260("COM6")
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    camera = UVCCamera(0, mtx, dist)
    camera.capture()

    # 棋盘格参数（根据你的棋盘格调整）
    pattern_size = (7, 5)  # 内角点数量 (列, 行) - 6行8列的内角点是7列5行
    print(f"棋盘格参数: {pattern_size[0]}x{pattern_size[1]} 内角点 (对应6行8列棋盘格)")
    print("如果不匹配，请修改代码中的 pattern_size\n")

    # 移动到观测位置 —— MyPalletizer 260 有 4 个关节 [J1, J2, J3, J4]
    # TODO: 请根据您的机械臂实际情况调整此角度
    observe_angles = [0.0, 0.0, 0.0, 0.0]
    print(f"移动到观测位置: {observe_angles}")
    mc.send_angles(observe_angles, 30)
    time.sleep(3)

    # 检测棋盘格
    print("检测棋盘格...")
    camera.update_frame()
    frame = camera.color_frame()

    ret, corners, display_frame = find_chessboard(frame, pattern_size)

    if not ret:
        print("❌ 未检测到棋盘格！")
        print("请确保：")
        print("  - 棋盘格完整在画面中")
        print("  - 光照充足")
        print("  - pattern_size 参数正确")
        cv2.imshow("Camera", frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        camera.release()
        return

    print(f"✓ 检测到棋盘格，共 {len(corners)} 个角点")
    cv2.imshow("Chessboard Detection", display_frame)
    cv2.waitKey(2000)

    # 选择4个角点用于标定
    corner_indices = [
        0,
        pattern_size[0] - 1,
        len(corners) - 1,
        len(corners) - pattern_size[0]
    ]

    corner_names = ["左上", "右上", "右下", "左下"]

    pixel_points = []
    robot_points = []

    print("\n" + "="*70)
    print("现在需要手动移动机械臂末端到4个角点位置")
    print("="*70 + "\n")

    # 释放伺服（MyPalletizer 260 有 4 个关节: J1-J4）
    print("释放伺服（J1-J4）...")
    for joint in range(1, 5):
        mc.release_servo(joint)
    print("✓ 伺服已释放\n")

    for i, (idx, name) in enumerate(zip(corner_indices, corner_names)):
        corner = corners[idx][0]
        pixel_points.append(corner)

        print(f"[{i+1}/4] {name}角点")
        print(f"  像素坐标: ({corner[0]:.1f}, {corner[1]:.1f})")

        display = frame.copy()
        cv2.circle(display, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), -1)
        cv2.putText(display, name, (int(corner[0])+15, int(corner[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Target Corner", display)
        cv2.waitKey(500)

        print(f"  请手动移动机械臂末端到{name}角点正上方")
        print(f"  移动到位后按 Enter 记录坐标")
        input("  按 Enter 继续: ")

        mc.power_on()
        time.sleep(1)

        coords = mc.get_coords()
        while coords is None:
            coords = mc.get_coords()
            time.sleep(0.1)

        robot_points.append([coords[0], coords[1]])
        print(f"  ✓ 机械臂坐标: ({coords[0]:.2f}, {coords[1]:.2f})")

        if i < 3:
            print("  释放伺服...")
            for joint in range(1, 5):
                mc.release_servo(joint)
        print()

    cv2.destroyAllWindows()

    print("="*70)
    print("计算映射矩阵...")

    pixel_points = np.array(pixel_points, dtype=np.float32)
    robot_points = np.array(robot_points, dtype=np.float32)

    H, _ = cv2.findHomography(pixel_points, robot_points)

    print("\n映射矩阵 (H):")
    print(H)

    print("\n验证映射精度:")
    errors = []
    for i, (pixel, robot, name) in enumerate(zip(pixel_points, robot_points, corner_names)):
        pixel_homo = np.array([pixel[0], pixel[1], 1])
        predicted_homo = H @ pixel_homo
        predicted = predicted_homo[:2] / predicted_homo[2]

        error = np.linalg.norm(predicted - robot)
        errors.append(error)
        print(f"  {name}: 误差 {error:.2f}mm")

    print(f"\n平均误差: {np.mean(errors):.2f}mm")
    print(f"最大误差: {np.max(errors):.2f}mm")

    calibration_data = {
        "homography_matrix": H.tolist(),
        "observe_angles": observe_angles,
        "observe_z": 180,
        "pixel_points": pixel_points.tolist(),
        "robot_points": robot_points.tolist(),
        "average_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "pattern_size": pattern_size
    }

    with open("plane_mapping.json", 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print("\n✓ 标定完成！")
    print("标定数据已保存到: plane_mapping.json")

    if np.mean(errors) < 5:
        print("✓ 标定精度优秀！")
    elif np.mean(errors) < 10:
        print("✓ 标定精度良好")
    else:
        print("⚠ 标定精度一般，建议重新标定")

    camera.release()

    print("\n" + "="*70)
    print("现在可以运行主程序：")
    print("  python math_solver_handwriting.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
