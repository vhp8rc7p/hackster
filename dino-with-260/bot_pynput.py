import time
import math
import mss
import pyautogui as gui
from pynput import keyboard

# --- CONFIGURATION ---
# 1. Scaling: MacBook Air = 2 (Retina). Standard = 1.
SCALE = 1 

# 2. Canvas Coordinates (Points) - Relative to the top-left of Screen 2
# canvas_x = 660
# canvas_y = 196
canvas_x, canvas_y = 434, 210
canvas_w = 600
canvas_h = 150

# 3. Search parameters (In Points, scaled later)
# y_search: vertical level to detect cactus (relative to canvas top)
y_search = 120
# y_search2: vertical level to detect birds
y_search2 = 100
# x_trigger: how many pixels in front of the Dino to look
x_trigger = 95
# x_look_ahead: how wide the detection beam is
x_look_ahead = 30

# --- INITIALIZATION ---
running = True
total_time = 0
last_time_step = 0

def on_press(key):
    global running
    try:
        if key.char == 'q':
            print("\n[!] Emergency stop. Exiting...")
            running = False
            return False
    except AttributeError:
        pass

# Start the 'q' key listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Bot starting in 3 seconds... Click into the Chrome window!")
time.sleep(3)

with mss.mss() as sct:
    # Identify Monitor 2 (Left Screen)
    if len(sct.monitors) < 3:
        print("Error: Monitor 2 not found. Check Display Settings.")
        exit()
    
    mon2 = sct.monitors[1]
    
    # Define the capture region in Global coordinates
    # We combine the monitor's offset with your local canvas coordinates
    monitor_region = {
        "left": mon2["left"] + canvas_x,
        "top": mon2["top"] + canvas_y,
        "width": canvas_w,
        "height": canvas_h
    }

    while running:
        t1 = time.time()
        
        # 1. Grab the canvas area
        sct_img = sct.grab(monitor_region)

        # 2. Get Background Color (Sampled from the top-right sky)
        # sct_img is already in physical pixels, so we scale the coordinate
        bg_pixel_x = 300
        bg_pixel_y = int(20 * SCALE)
        bgColor = sct_img.pixel(bg_pixel_x, bg_pixel_y)
        print(f"Background color sampled at ({bg_pixel_x}, {bg_pixel_y}): {bgColor}    ", end="\r")
        # 3. Simulate Acceleration
        if math.floor(total_time) != last_time_step:
            # Gradually increase look-ahead as the game speeds up
            x_look_ahead += 1 
            last_time_step = math.floor(total_time)

        # 4. Define Search Range (Converted to physical pixels)
        p_start = int(x_trigger * SCALE)
        p_end = int((x_trigger + x_look_ahead) * SCALE)
        p_y1 = int(y_search * SCALE)
        p_y2 = int(y_search2 * SCALE)

        # 5. Scan for obstacles
        # We scan from the front of the dino to the end of the beam
        for i in range(p_start, p_end):
            # If the pixel color isn't the background, it's an obstacle
            if sct_img.pixel(i, p_y1) != bgColor or \
               sct_img.pixel(i, p_y2) != bgColor:
                gui.press('space')
                break

        # Timing for speed logic
        total_time += (time.time() - t1)

listener.stop()
print("Bot deactivated.")