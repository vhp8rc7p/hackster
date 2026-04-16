import cv2
import numpy as np
import time
import argparse
from pymycobot import MyPalletizer260
from pump import set_pump

# --- 1. 初始化 ---
# try:
#     arm = MyPalletizer260("COM4", 115200)
#     print("机器人连接成功。")
# except Exception as e:
#     print(f"机器人连接失败: {e}")
#     exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开相机")
    exit()
h, w, _ = cap.read()[1].shape
u_center, v_center = w / 2, h / 2
print("相机初始化成功。")

# --- 2. 参数 ---
# Color Ranges (HSV)
COLOR_RANGES = {
    "red": ([0, 70, 50], [10, 255, 255]),
    "red2": ([170, 70, 50], [180, 255, 255]),
    "green": ([35, 43, 46], [77, 255, 255]) # NOTE: This may need tuning
}
# Robot Parameters
DIR_X = -1
DIR_Y = -1
SAFE_Z = 200
PICK_Z = 75
DROP_COORDS = [0, -150, 100]
MOVE_SPEED = 40
TOOL_OFFSET_X = -10
TOOL_OFFSET_Y = -40

# --- 3. 加载标定矩阵 ---
try:
    M = np.load("calibration_matrix.npy")
    R_pixel_to_mm = M[:, :2]
    print("成功加载标定矩阵。")
except FileNotFoundError:
    print("错误：找不到 calibration_matrix.npy。")
    exit()

# --- 4. 功能函数 ---
def find_all_cubes(frame):
    """Finds all colored cubes in the frame and returns a list of dictionaries."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    found_cubes = []
    display_frame = frame.copy() # Create a copy for drawing

    # Combine red masks
    red_mask1 = cv2.inRange(blurred, np.array(COLOR_RANGES["red"][0]), np.array(COLOR_RANGES["red"][1]))
    red_mask2 = cv2.inRange(blurred, np.array(COLOR_RANGES["red2"][0]), np.array(COLOR_RANGES["red2"][1]))
    full_red_mask = red_mask1 + red_mask2
    kernel = np.ones((5,5), np.uint8)
    full_red_mask = cv2.morphologyEx(full_red_mask, cv2.MORPH_CLOSE, kernel)
    # Process red mask
    mask = cv2.morphologyEx(full_red_mask, cv2.MORPH_OPEN, kernel)

# Fill small holes INSIDE the cube, like glare (Closing)
    red_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                found_cubes.append({"color": "red", "coords": (u, v)})
                # Draw on the display frame
                cv2.drawContours(display_frame, [c], -1, (0, 0, 255), 2)
                cv2.putText(display_frame, "RED", (u, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Process green mask
    green_mask = cv2.inRange(hsv, np.array(COLOR_RANGES["green"][0]), np.array(COLOR_RANGES["green"][1]))
    green_mask = cv2.erode(green_mask, None, iterations=2)
    green_mask = cv2.dilate(green_mask, None, iterations=2)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])
                found_cubes.append({"color": "green", "coords": (u, v)})
                # Draw on the display frame
                cv2.drawContours(display_frame, [c], -1, (0, 255, 0), 2)
                cv2.putText(display_frame, "GREEN", (u, v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return found_cubes, display_frame


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
    # This function will be called by the agent with the user's choice
    pass # The main logic will be outside this function for interaction

if __name__ == '__main__':
    # The script is designed to be controlled by the agent,
    # so the main execution block is separated for clarity.
    # Step 1: Scan for cubes
    print("移动到初始观察位置...")
    # arm.send_angles([0, 0, 0, 0], MOVE_SPEED)
    time.sleep(2.5)
    # set_pump(arm, 3, False)
    time.sleep(1)

    print("扫描桌面上的方块...")
    # Add a loop for continuous scanning display
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机画面。")
            break

        detected_cubes, display_frame = find_all_cubes(frame)

        # Display the frame
        cv2.imshow("Cube Detection", display_frame)

        # Check for a key press to confirm the scan
        key = cv2.waitKey(100) & 0xFF
        if key == ord('c'): # Press 'c' to confirm scan and continue
            break
        elif key == ord('q'): # Press 'q' to quit
            detected_cubes = [] # Empty the list if quitting
            break

    if not detected_cubes:
        print("桌面上未发现任何红色或绿色方块。")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Step 2: Report and get user choice (This part is handled by the agent)
    # For standalone testing, you would add an input() prompt here.
    # The agent will call the execution part with the chosen cube.
    print("\n--- DETECTION COMPLETE ---")
    print("Detected cubes:")
    for cube in detected_cubes:
        print(f"  - A {cube['color']} cube at pixel coordinates {cube['coords']}")

    # The agent will now take over, ask the user, and then call the execution function.
    # This script will now exit, waiting for the agent's next command.
    cap.release()
    cv2.destroyAllWindows()
