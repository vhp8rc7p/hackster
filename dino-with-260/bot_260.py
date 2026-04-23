import time
import math
import mss
from pynput import keyboard
from pymycobot import MyPalletizer260
mc=MyPalletizer260("/dev/tty.usbserial-59010016581",115200)


# Your provided coordinates
# Point 1 (Z=24.4) -> Likely the "Press" (Down)
# Point 2 (Z=31.1) -> Likely the "Release" (Up)
Z_RELEASED = [187.7, 11.2, 22, 9.4]
Z_PRESSED = [186.3, 11.1, 18.2, 7.64]
# Move to starting position
print("Moving arm to Hover position...")
mc.send_coords(Z_RELEASED, 60) 
time.sleep(2)

# --- GAME CONFIGURATION ---
SCALE = 1 
# canvas_x, canvas_y = 660, 196
canvas_x, canvas_y = 434, 210
canvas_w, canvas_h = 600, 150

y_search = 120
y_search2 = 100

# INCREASED TRIGGER: Robot arms are slower than software. 
# You need to detect obstacles much further away.
x_trigger = 95 # Adjusted from 95 to account for arm travel time
x_look_ahead = 30

running = True
total_time = 0
last_time_step = 0

def on_press(key):
    global running
    try:
        if key.char == 'q':
            running = False
            return False
    except AttributeError: pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Bot starting in 3s... Ensure arm has clear path!")
time.sleep(3)

with mss.mss() as sct:
    mon2 = sct.monitors[1]
    monitor_region = {
        "left": mon2["left"] + canvas_x,
        "top": mon2["top"] + canvas_y,
        "width": canvas_w,
        "height": canvas_h
    }

    while running:
        t1 = time.time()
        sct_img = sct.grab(monitor_region)

        # Sampling background
        bg_pixel_x, bg_pixel_y = 300, int(20 * SCALE)
        bgColor = sct_img.pixel(bg_pixel_x, bg_pixel_y)

        # Acceleration Logic
        if math.floor(total_time) != last_time_step:
            x_look_ahead += 1 
            last_time_step = math.floor(total_time)

        p_start = int(x_trigger * SCALE)
        p_end = int((x_trigger + x_look_ahead) * SCALE)
        p_y1, p_y2 = int(y_search * SCALE), int(y_search2 * SCALE)

        # Scan for obstacles
        for i in range(p_start, p_end):
            if sct_img.pixel(i, p_y1) != bgColor or sct_img.pixel(i, p_y2) != bgColor:
                # --- PHYSICAL JUMP ---
                # Speed 100 is max. Mode 0 is linear interpolation.
                mc.send_coords(Z_PRESSED, 60)
                time.sleep(0.1)
                mc.send_coords(Z_RELEASED,60)
                
                # Small cool-down to prevent the arm from stuttering 
                # or double-jumping on the same cactus
                time.sleep(0.3) 
                break

        total_time += (time.time() - t1)

listener.stop()
print("Bot deactivated. Arm returning to safety.")