"""
B区域平面映射标定（MyPalletizer 260 版本）
使用棋盘格标定B区域的像素坐标到机械臂坐标的映射关系
"""
import cv2
import numpy as np
from uvc_camera import UVCCamera
from pymycobot import MyPalletizer260
import json
import time

# 初始化机械臂（MyPalletizer 260）
mc = MyPalletizer260("COM6")

# 加载相机参数
camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]

# 初始化相机
camera = UVCCamera(0, mtx, dist)

# 棋盘格参数（6行8列，内角点7×5）
PATTERN_SIZE = (7, 5)
SQUARE_SIZE = 20  # mm

# B区域观测位置（从配置文件读取）
try:
    with open("math_solver_config.json", 'r') as f:
        config = json.load(f)
        OBSERVE_ANGLES_B = config["POSITION_B"]
except:
    print("ERROR: 无法读取 math_solver_config.json")
    print("请确保配置文件存在且包含 POSITION_B")
    exit()

OBSERVE_Z = 200  # mm

print("\n" + "="*70)
print("B区域平面映射标定 (MyPalletizer 260)")
print("="*70)
print(f"\n棋盘格规格: {PATTERN_SIZE[0]}×{PATTERN_SIZE[1]} 内角点")
print(f"方格大小: {SQUARE_SIZE}mm")
print(f"观测位置: {OBSERVE_ANGLES_B}")
print(f"观测高度: Z={OBSERVE_Z}mm")
print("\n" + "="*70 + "\n")

def wait_move():
    time.sleep(0.5)
    while mc.is_moving() == 1:
        time.sleep(0.2)

def main():
    print("步骤1: 移动到B区域观测位置")
    mc.send_angles(OBSERVE_ANGLES_B, 30)
    wait_move()

    print(f"步骤2: 调整到观测高度 Z={OBSERVE_Z}mm")
    current_coords = mc.get_coords()
    while current_coords is None:
        current_coords = mc.get_coords()
        time.sleep(0.1)

    if current_coords[2] != OBSERVE_Z:
        current_coords[2] = OBSERVE_Z
        mc.send_coords(current_coords, 30)
        wait_move()

    print("✓ 到达观测位置\n")

    camera.capture()

    print("步骤3: 检测棋盘格角点")
    print("请将棋盘格放置在B区域，确保完全可见")
    print("按 's' 键开始标定，按 'q' 键退出\n")

    pixel_points = []
    robot_points = []

    while True:
        camera.update_frame()
        frame = camera.color_frame()

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

        display = frame.copy()

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(display, PATTERN_SIZE, corners_refined, ret)

            cv2.putText(display, "Chessboard detected! Press 's' to calibrate",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Chessboard not found",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("B Area Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n标定已取消")
            cv2.destroyAllWindows()
            camera.release()
            return

        elif key == ord('s') and ret:
            print("\n开始标定...")

            corner_indices = [
                0,
                PATTERN_SIZE[0] - 1,
                PATTERN_SIZE[0] * PATTERN_SIZE[1] - 1,
                PATTERN_SIZE[0] * (PATTERN_SIZE[1] - 1)
            ]

            for i, idx in enumerate(corner_indices):
                corner_names = ["左上", "右上", "右下", "左下"]
                pixel_x, pixel_y = corners_refined[idx][0]

                print(f"\n角点 {i+1}/4 ({corner_names[i]})")
                print(f"  像素坐标: ({pixel_x:.2f}, {pixel_y:.2f})")

                highlight = frame.copy()
                cv2.drawChessboardCorners(highlight, PATTERN_SIZE, corners_refined, ret)
                cv2.circle(highlight, (int(pixel_x), int(pixel_y)), 15, (0, 0, 255), 3)
                cv2.putText(highlight, corner_names[i], (int(pixel_x) + 20, int(pixel_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("B Area Calibration", highlight)
                cv2.waitKey(500)

                current_coords = mc.get_coords()
                while current_coords is None:
                    current_coords = mc.get_coords()
                    time.sleep(0.1)

                robot_x, robot_y = current_coords[0], current_coords[1]

                print(f"  当前机械臂坐标: ({robot_x:.2f}, {robot_y:.2f})")

                print(f"  释放机械臂...")
                mc.release_all_servos()
                time.sleep(0.5)

                print(f"  请手动移动机械臂，使末端对准 {corner_names[i]} 角点")
                print(f"  移动完成后按 Enter 键确认...")

                input()

                new_coords = mc.get_coords()
                while new_coords is None:
                    new_coords = mc.get_coords()
                    time.sleep(0.1)

                robot_x, robot_y = new_coords[0], new_coords[1]

                print(f"  ✓ 记录坐标: 像素({pixel_x:.2f}, {pixel_y:.2f}) -> 机械臂({robot_x:.2f}, {robot_y:.2f})")

                pixel_points.append([pixel_x, pixel_y])
                robot_points.append([robot_x, robot_y])

            print("\n步骤4: 计算单应性矩阵...")
            pixel_points_np = np.array(pixel_points, dtype=np.float32)
            robot_points_np = np.array(robot_points, dtype=np.float32)

            H, status = cv2.findHomography(pixel_points_np, robot_points_np)

            print("✓ 单应性矩阵计算完成")
            print("\n单应性矩阵 H:")
            print(H)

            print("\n步骤5: 验证标定精度...")
            errors = []
            for i, (pixel_pt, robot_pt) in enumerate(zip(pixel_points, robot_points)):
                pixel_homo = np.array([pixel_pt[0], pixel_pt[1], 1])
                robot_homo = H @ pixel_homo
                robot_predicted = robot_homo[:2] / robot_homo[2]

                error = np.linalg.norm(robot_predicted - robot_pt)
                errors.append(error)

                print(f"  角点 {i+1}: 误差 = {error:.2f}mm")

            avg_error = np.mean(errors)
            max_error = np.max(errors)

            print(f"\n平均误差: {avg_error:.2f}mm")
            print(f"最大误差: {max_error:.2f}mm")

            if avg_error < 5:
                print("✓ 标定精度良好")
            elif avg_error < 10:
                print("⚠ 标定精度一般，建议重新标定")
            else:
                print("❌ 标定精度较差，请重新标定")

            print("\n步骤6: 保存标定数据...")

            pixel_points_list = [[float(x), float(y)] for x, y in pixel_points]
            robot_points_list = [[float(x), float(y)] for x, y in robot_points]

            calib_data = {
                "homography_matrix": H.tolist(),
                "observe_angles": OBSERVE_ANGLES_B,
                "observe_z": OBSERVE_Z,
                "pixel_points": pixel_points_list,
                "robot_points": robot_points_list,
                "average_error": float(avg_error),
                "max_error": float(max_error),
                "pattern_size": list(PATTERN_SIZE)
            }

            with open("plane_mapping_B.json", 'w') as f:
                json.dump(calib_data, f, indent=2)

            print("✓ 标定数据已保存到 plane_mapping_B.json")

            print("\n" + "="*70)
            print("B区域标定完成！")
            print("="*70)

            break

    cv2.destroyAllWindows()
    camera.release()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n标定已中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
