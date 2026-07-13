"""
基于PaddleOCR的数学求解器（MyPalletizer 260 版本）
使用PaddleOCR进行数字和运算符识别
"""
import cv2
from uvc_camera import UVCCamera
import numpy as np
import json
import time
from pymycobot import MyPalletizer260
from paddleocr import PaddleOCR

# 初始化PaddleOCR
print("初始化 PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)
print("✓ PaddleOCR 初始化完成")

def detect_and_recognize(frame, confidence_threshold=0.5):
    """使用PaddleOCR检测并识别画面中的字符

    Args:
        frame: 输入图像
        confidence_threshold: 置信度阈值，默认0.5

    Returns:
        检测结果列表，按X坐标排序
    """
    result = ocr.ocr(frame, cls=True)

    detections = []

    if result and result[0]:
        for line in result[0]:
            box = line[0]
            text = line[1][0].strip()
            confidence = line[1][1]

            if confidence < confidence_threshold:
                continue

            if not text:
                continue

            # 计算中心点
            pts = np.array(box, dtype=np.int32)
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            # 统一运算符
            if text in ['×', 'x', 'X', '*']:
                text = '*'
            elif text == '÷':
                text = '/'
            elif text in ['一', '-']:
                text = '-'
            elif text in ['十', '+']:
                text = '+'

            # 只保留数字和运算符
            if text.isdigit() or text in '+-*/':
                detections.append({
                    'char': text,
                    'confidence': confidence,
                    'center': (center_x, center_y)
                })

    # 按X坐标排序（从左到右）
    detections.sort(key=lambda d: d['center'][0])
    return detections

# ========== 机械臂控制 ==========

# MyPalletizer 260 通常使用 COM 端口 (Windows) 或 /dev/ttyUSB0 / /dev/ttyAMA0 (Linux)
mc = MyPalletizer260("COM6")

camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]

# 加载平面映射标定数据（A区域）
try:
    with open("plane_mapping.json", 'r') as f:
        calib_data_A = json.load(f)
    H_A = np.array(calib_data_A["homography_matrix"])
    observe_z_A = calib_data_A.get("observe_z", 200)
    print("✓ A区域 plane mapping calibration loaded")
    print(f"  Average error: {calib_data_A['average_error']:.2f}mm")
except FileNotFoundError:
    print("ERROR: plane_mapping.json not found!")
    print("Please run: python calibrate_plane_mapping.py")
    exit()

# 加载B区域平面映射标定数据
try:
    with open("plane_mapping_B.json", 'r') as f:
        calib_data_B = json.load(f)
    H_B = np.array(calib_data_B["homography_matrix"])
    observe_z_B = calib_data_B.get("observe_z", 200)
    print("✓ B区域 plane mapping calibration loaded")
    print(f"  Average error: {calib_data_B['average_error']:.2f}mm")
except FileNotFoundError:
    print("WARNING: plane_mapping_B.json not found!")
    print("B区域将使用A区域的标定数据（可能不准确）")
    print("建议运行: python calibrate_plane_mapping_B.py")
    H_B = H_A
    observe_z_B = observe_z_A

# 初始化相机但不立即打开
camera = UVCCamera(0, mtx, dist)

try:
    with open("math_solver_config.json", 'r') as f:
        config = json.load(f)
        POSITION_A = config["POSITION_A"]
        POSITION_B = config["POSITION_B"]  # B区域位置
        POSITION_PLACE_TRANSITION = config.get("POSITION_PLACE_TRANSITION")
        POSITION_PLACE_FINAL = config.get("POSITION_PLACE_FINAL")
except:
    print("ERROR: math_solver_config.json not found!")
    exit()

def wait_move():
    time.sleep(0.5)
    while mc.is_moving() == 1:
        time.sleep(0.2)

# ========== 辅助函数：带超时的移动等待 ==========
def wait_move_safe(timeout=10):
    """等待机械臂移动完成，带超时保护"""
    start_time = time.time()
    while mc.is_moving() == 1:
        if time.time() - start_time > timeout:
            print(f"⚠️ 等待移动超时 ({timeout}秒)，强制继续")
            break
        time.sleep(0.2)

def pixel_to_robot(pixel_x, pixel_y, H):
    """使用平面映射矩阵将像素坐标转换为机械臂坐标"""
    pixel_homo = np.array([pixel_x, pixel_y, 1])
    robot_homo = H @ pixel_homo
    robot_coords = robot_homo[:2] / robot_homo[2]
    return robot_coords[0], robot_coords[1]

def pump_on():
    mc.set_basic_output(5, 0)
    time.sleep(0.05)

def pump_off():
    mc.set_basic_output(5, 1)
    time.sleep(0.05)
    mc.set_basic_output(2, 0)
    time.sleep(1)
    mc.set_basic_output(2, 1)
    time.sleep(0.05)

def pick_digit(target_char):
    """使用B区域标定抓取指定数字"""
    print(f"\n{'='*60}")
    print(f"抓取数字: {target_char}")
    print(f"{'='*60}\n")

    target_char = str(target_char)

    # 1. 移动到B区域观测位置
    print("1. 移动到B区域观测位置...")
    mc.send_angles(POSITION_B, 30)
    wait_move_safe(10)
    print("✓ 到达B区域\n")

    # 2. OCR识别答案数字
    print(f"2. 识别数字 '{target_char}'...")
    target_det = None
    max_attempts = 15

    for attempt in range(max_attempts):
        camera.update_frame()
        frame = camera.color_frame()

        if frame is None:
            time.sleep(0.2)
            continue

        if attempt % 3 == 0:
            print(f"  尝试 {attempt + 1}/{max_attempts}...")

        result = ocr.ocr(frame, cls=True)

        if result and result[0]:
            for line in result[0]:
                text = line[1][0].strip()
                confidence = line[1][1]

                if text == target_char and confidence > 0.5:
                    box = line[0]
                    pts = np.array(box, dtype=np.int32)
                    center_x = int(np.mean(pts[:, 0]))
                    center_y = int(np.mean(pts[:, 1]))

                    target_det = {
                        'char': text,
                        'confidence': confidence,
                        'center': (center_x, center_y)
                    }
                    print(f"  ✓ 找到数字 '{target_char}' (置信度: {confidence:.2f})")
                    print(f"  像素坐标: {target_det['center']}")
                    break

        if target_det:
            break

        time.sleep(0.2)

    if target_det is None:
        print(f"\n❌ 未找到数字 '{target_char}'")
        return False

    # 3. 像素坐标 → 机械臂坐标（使用B区域标定）
    pixel_x, pixel_y = target_det['center']
    robot_x, robot_y = pixel_to_robot(pixel_x, pixel_y, H_B)
    print(f"\n3. 坐标转换:")
    print(f"  像素坐标: ({pixel_x}, {pixel_y})")
    print(f"  机械臂坐标: ({robot_x:.1f}, {robot_y:.1f})")

    # 添加偏移
    offset_x = 5
    offset_y = -10
    robot_x_adjusted = robot_x + offset_x
    robot_y_adjusted = robot_y + offset_y
    print(f"  偏移后坐标: ({robot_x_adjusted:.1f}, {robot_y_adjusted:.1f})")

    # 4. 移动到目标上方（保持观测高度）
    # MyPalletizer 260 的坐标是 [x, y, z, θ]（4个值）
    print(f"\n4. 移动到目标上方...")
    current_coords = mc.get_coords()
    while current_coords is None:
        current_coords = mc.get_coords()
        time.sleep(0.1)

    target_above = current_coords.copy()
    target_above[0] = robot_x_adjusted
    target_above[1] = robot_y_adjusted
    target_above[2] = observe_z_B

    print(f"  目标坐标: {target_above}")
    mc.send_coords(target_above, 30)
    wait_move_safe(10)
    print("✓ 到达目标上方")

    # 5. 下降到Z=70mm
    print(f"\n5. 下降到Z=70mm...")
    target_down = target_above.copy()
    target_down[2] = 70

    print(f"  目标坐标: {target_down}")
    mc.send_coords(target_down, 20)
    time.sleep(5)  # 强制等待
    wait_move_safe(10)

    final_coords = mc.get_coords()
    if final_coords:
        print(f"  最终高度: {final_coords[2]:.1f}mm")
        if abs(final_coords[2] - 70) < 5:
            print("✓ 成功下降到抓取高度")
        else:
            print(f"⚠️ 高度偏差较大，期望70mm，实际{final_coords[2]:.1f}mm")

    # 6. 启动吸泵
    print("\n6. 启动吸泵...")
    pump_on()
    time.sleep(1)
    print("✓ 吸泵已启动")

    # 7. 上升
    print("\n7. 上升...")
    mc.send_coords(target_above, 20)
    wait_move_safe(10)
    print("✓ 已上升")

    print(f"\n{'='*60}")
    print(f"✓ 成功抓取数字 '{target_char}'")
    print(f"{'='*60}\n")

    return True



def place_digit():
    """放置数字到指定位置"""
    print(f"\n{'='*60}")
    print("放置数字")
    print(f"{'='*60}\n")

    # 从配置文件读取放置位置（4元素坐标: [x, y, z, θ]）
    # TODO: 请在 math_solver_config.json 中根据实际情况标定 PLACE_ABOVE / PLACE_DOWN
    place_above = config.get("PLACE_ABOVE", [209.3, -182.0, 151.8, 0.0])
    place_down = config.get("PLACE_DOWN", [211.7, -185.3, 68.1, 0.0])

    # 1. 移动到放置点上方
    print(f"移动到放置点上方...")
    print(f"  目标坐标: {place_above}")
    mc.send_coords(place_above, 20)
    time.sleep(3)
    wait_move_safe(10)
    print("✓ 到达放置点上方")

    # 2. 下降到放置高度
    print(f"\n下降到放置高度...")
    print(f"  目标坐标: {place_down}")
    mc.send_coords(place_down, 20)
    time.sleep(5)
    wait_move_safe(10)

    # 验证最终位置
    final_coords = mc.get_coords()
    if final_coords:
        print(f"  最终坐标: {final_coords}")
        print(f"  最终高度: {final_coords[2]:.1f}mm")
    print("✓ 到达放置高度")

    # 3. 释放吸泵
    print("\n释放吸泵...")
    pump_off()
    time.sleep(1)
    print("✓ 已释放")

    # 4. 上升
    print("\n上升...")
    target_coords_up = final_coords.copy()
    target_coords_up[2] = 150
    mc.send_coords(target_coords_up, 20)
    time.sleep(5)
    while mc.is_moving():
        time.sleep(0.5)
    print("✓ 已上升")

    print(f"\n{'='*60}")
    print("✓ 放置完成")
    print(f"{'='*60}\n")



# ========== 主流程 ==========

def evaluate_expression(tokens):
    """计算表达式，支持运算符优先级（先乘除后加减）"""
    if len(tokens) == 0:
        return None

    if len(tokens) == 1:
        try:
            return int(tokens[0])
        except:
            return None

    numbers = []
    operators = []

    for i, token in enumerate(tokens):
        if i % 2 == 0:
            try:
                numbers.append(int(token))
            except:
                print(f"ERROR: 无效的数字 '{token}'")
                return None
        else:
            if token in '+-*/':
                operators.append(token)
            else:
                print(f"ERROR: 无效的运算符 '{token}'")
                return None

    if len(numbers) != len(operators) + 1:
        print(f"ERROR: 表达式格式错误，数字{len(numbers)}个，运算符{len(operators)}个")
        return None

    # 先乘除
    i = 0
    while i < len(operators):
        if operators[i] in '*/':
            op = operators[i]
            left = numbers[i]
            right = numbers[i + 1]

            if op == '*':
                result = left * right
            else:
                if right == 0:
                    print("ERROR: 除数不能为0")
                    return None
                result = left // right

            numbers[i] = result
            numbers.pop(i + 1)
            operators.pop(i)
        else:
            i += 1

    # 后加减
    i = 0
    while i < len(operators):
        op = operators[i]
        left = numbers[i]
        right = numbers[i + 1]

        if op == '+':
            result = left + right
        else:
            result = left - right

        numbers[i] = result
        numbers.pop(i + 1)
        operators.pop(i)

    return numbers[0]

def parse_equation(detections):
    """解析检测结果为算式（支持运算符优先级）"""
    tokens = [d['char'] for d in detections]

    print(f"识别结果: {tokens}")

    if len(tokens) >= 2:
        has_operator = any(t in '+-*/' for t in tokens)
        if not has_operator:
            new_tokens = []
            for i, token in enumerate(tokens):
                new_tokens.append(token)
                if i < len(tokens) - 1:
                    new_tokens.append('*')
            tokens = new_tokens
            print("  未检测到运算符，默认使用乘法")

    equation_str = ''.join(tokens)
    print(f"算式: {equation_str}")

    return tokens, equation_str

def main():
    print("\n" + "="*70)
    print("数学求解器 - 使用平面映射精确抓取 (MyPalletizer 260)")
    print("="*70 + "\n")

    camera.capture()

    print("步骤1: 移动到A区域识别算式")
    mc.send_angles(POSITION_A, 50)
    wait_move()
    print("等待稳定...")
    time.sleep(2)

    print("识别算式...")
    camera.update_frame()
    frame = camera.color_frame()

    if frame is None:
        print("ERROR: 无法获取图像")
        return

    cv2.imshow("Area A - Equation", frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    results = detect_and_recognize(frame, confidence_threshold=0.5)

    if len(results) == 0:
        print("ERROR: 未检测到字符")
        return

    print(f"检测到 {len(results)} 个字符:")
    for r in results:
        print(f"  '{r['char']}' (置信度: {r['confidence']:.2f})")

    tokens, equation_str = parse_equation(results)

    if tokens is None or equation_str is None:
        print("ERROR: 无法解析算式")
        return

    answer = evaluate_expression(tokens)

    if answer is None:
        print("ERROR: 无法计算表达式")
        return

    print(f"\n算式: {equation_str} = {answer}\n")

    print("步骤2: 抓取答案")
    success = pick_digit(answer)

    if not success:
        print(f"ERROR: 抓取数字 {answer} 失败")
        return

    print("步骤3: 放置数字")
    place_digit()

    print("步骤4: 回到A区域")
    mc.send_angles(POSITION_A, 50)
    wait_move()

    print("\n" + "="*70)
    print("任务完成！")
    print(f"结果: {equation_str} = {answer}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pump_off()
        cv2.destroyAllWindows()
        camera.release()
