import mss
from PIL import Image, ImageDraw

def capture_mon_with_fine_grid(index=1,step=50):
    with mss.mss() as sct:
        # Check if monitor 2 actually exists
        # if len(sct.monitors) < 3:
        #     print("Monitor 2 not found! Using Monitor 1.")
        #     mon = sct.monitors[1]
        # else:
        #     mon = sct.monitors[2]
        mon = sct.monitors[index]
        print(f"Capturing monitor at: {mon}")
        sct_img = sct.grab(mon)

        # Convert the mss buffer to a PIL Image for drawing
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        draw = ImageDraw.Draw(img)
        
        width, height = img.size
        
        # Colors: Red for major (100px), Grey for minor (50px), Yellow for text
        major_c = (255, 0, 0)
        minor_c = (80, 80, 80)
        text_c = (255, 255, 0)

        # Draw Vertical Lines (X-axis)
        for x in range(0, width, step):
            color = major_c if x % 100 == 0 else minor_c
            draw.line([(x, 0), (x, height)], fill=color, width=1)
            if x % 100 == 0:
                draw.text((x + 2, 5), str(x), fill=text_c)

        # Draw Horizontal Lines (Y-axis)
        for y in range(0, height, step):
            color = major_c if y % 100 == 0 else minor_c
            draw.line([(0, y), (width, y)], fill=color, width=1)
            if y % 100 == 0:
                draw.text((5, y + 2), str(y), fill=text_c)

        # Save it
        img.save("mon2_grid_map.png")
        print(f"Captured {width}x{height} pixels. File saved as mon2_grid_map.png")

if __name__ == "__main__":
    capture_mon_with_fine_grid(1)