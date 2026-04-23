import mss
from PIL import Image, ImageDraw # Added ImageDraw
import os

# --- 1. GAME CANVAS LOCATION (POINTS) ---
# canvas_x = 660
# canvas_y = 196
canvas_x = 434
canvas_y = 210
canvas_w = 600
canvas_h = 150

# --- 2. MAC RETINA SCALING ---
SCALE = 1 # MacBook Air uses 2. Use 1 for standard screens.

# --- 3. BOT SEARCH PARAMETERS (POINTS - relative to canvas) ---
# Use the exact numbers from your final bot code here.
x_trigger = 95
x_look_ahead = 45 # The initial width of the beam
y_cactus = 115 
y_bird = 75

# --- THE SCRIPT ---
def verify_and_visualize_beam(index):
    with mss.mss() as sct:
        # 1. Identify Monitor 2 (Left Screen)
        if len(sct.monitors) < 3:
            print("Error: Monitor 2 not found.")
            return

        mon2 = sct.monitors[index]

        # 2. Define the capture region (Logical)
        monitor_region = {
            "left": mon2["left"] + canvas_x,
            "top": mon2["top"] + canvas_y,
            "width": canvas_w,
            "height": canvas_h
        }
        
        # 3. Grab the region (results in physical pixels)
        try:
            sct_img = sct.grab(monitor_region)
        except Exception as e:
            print(f"Capture failed: {e}")
            return

        # 4. Convert to a PIL Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # --- NEW: DRAW THE VISUALIZATIONS ---
        # Initialize the draw object
        draw = ImageDraw.Draw(img)

        # A. Calculate Physical Pixel Coordinates
        p_x_start = int(x_trigger * SCALE)
        p_x_end = int((x_trigger + x_look_ahead) * SCALE)
        p_y_cactus = int(y_cactus * SCALE)
        p_y_bird = int(y_bird * SCALE)

        # B. Define Colors (RGBA - using Alpha for transparency)
        green = (0, 255, 0, 200)   # Cactus Search line
        cyan = (0, 255, 255, 200)   # Bird Search line
        red_box = (255, 0, 0, 100) # Semi-transparent area box
        
        # C. Draw the visualization box (The Beam Area)
        # Note: The 'rectangle' function works in PIL, 
        # but transparency requires a slightly different approach 
        # (overlaying an image). For now, we draw outlines for clarity.
        
        # Top Beam (Birds)
        draw.line([(p_x_start, p_y_bird), (p_x_end, p_y_bird)], fill=cyan, width=2)
        # Bottom Beam (Cactus)
        draw.line([(p_x_start, p_y_cactus), (p_x_end, p_y_cactus)], fill=green, width=2)
        
        # Draw vertical lines for start/end of beam
        draw.line([(p_x_start, 0), (p_x_start, sct_img.size[1])], fill=(255, 255, 0), width=1) # Yellow Start
        draw.line([(p_x_end, 0), (p_x_end, sct_img.size[1])], fill=(255, 0, 0), width=1)   # Red End

        # --- SAVE THE IMAGE ---
        filename = "dino_beam_visualization.png"
        img.save(filename)
        
        print("-" * 40)
        print(f"SUCCESS! Image saved as: {filename}")
        print("-" * 40)
        print("HOW TO READ THE MAP:")
        print("1. Yellow vertical line: x_trigger (When to start looking).")
        print("2. Red vertical line: x_look_ahead end (How far ahead to look).")
        print("3. Cyan horizontal line: Bird Search level.")
        print("4. Green horizontal line: Cactus Search level.")
        print("-" * 40)

if __name__ == "__main__":
    verify_and_visualize_beam(1)