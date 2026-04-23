import mss
import mss.tools
from PIL import Image, ImageDraw, ImageFont

def capture_and_grid(step=200):
    with mss.mss() as sct:
        # Monitor 0 is the "all-in-one" virtual desktop across all screens
        all_monitors = sct.monitors[0]
        
        print(f"Capturing virtual desktop: {all_monitors}")
        screenshot = sct.grab(all_monitors)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        draw = ImageDraw.Draw(img)
        
        # Determine the boundaries
        width, height = img.size
        left = all_monitors["left"]
        top = all_monitors["top"]

        # Define colors
        grid_color = (255, 0, 0, 128)  # Red lines
        text_color = (255, 255, 0)      # Yellow text
        
        # Draw Vertical Lines (X-axis)
        # We calculate based on global coordinates
        start_x = (left // step) * step
        for x_val in range(start_x, left + width, step):
            # Convert global X to local pixel X
            local_x = x_val - left
            draw.line([(local_x, 0), (local_x, height)], fill=grid_color, width=1)
            draw.text((local_x + 5, 10), f"X: {x_val}", fill=text_color)

        # Draw Horizontal Lines (Y-axis)
        start_y = (top // step) * step
        for y_val in range(start_y, top + height, step):
            # Convert global Y to local pixel Y
            local_y = y_val - top
            draw.line([(0, local_y), (width, local_y)], fill=grid_color, width=1)
            draw.text((10, local_y + 5), f"Y: {y_val}", fill=text_color)

        # Draw a thick cross at (0,0) - your Primary Monitor origin
        origin_x = 0 - left
        origin_y = 0 - top
        if 0 <= origin_x <= width and 0 <= origin_y <= height:
            draw.line([(origin_x, 0), (origin_x, height)], fill=(0, 255, 0), width=5)
            draw.line([(0, origin_y), (width, origin_y)], fill=(0, 255, 0), width=5)
            draw.text((origin_x + 10, origin_y + 10), "PRIMARY ORIGIN (0,0)", fill=(0, 255, 0))

        # Save the result
        output = "desktop_grid_map.png"
        img.save(output)
        print(f"Success! Map saved as {output}")

if __name__ == "__main__":
    capture_and_grid(step=200) # Grid every 200 pixels