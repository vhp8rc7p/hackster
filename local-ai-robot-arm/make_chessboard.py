"""
Generates a chessboard PNG for hand-eye calibration.
10 cols x 7 rows of squares -> 9 x 6 INNER corners (what cv2.findChessboardCorners wants).
20 mm squares at 600 DPI, with a 10 mm white border so corner detection has margin.
"""
import numpy as np
import cv2

MM_PER_IN = 25.4
DPI = 600
SQUARE_MM = 20.0
COLS = 10   # squares across  -> 9 inner corners across
ROWS = 7    # squares down     -> 6 inner corners down
BORDER_MM = 10.0

sq_px = int(round(SQUARE_MM / MM_PER_IN * DPI))
bd_px = int(round(BORDER_MM / MM_PER_IN * DPI))

board_w = COLS * sq_px
board_h = ROWS * sq_px
img_w = board_w + 2 * bd_px
img_h = board_h + 2 * bd_px

img = np.full((img_h, img_w), 255, np.uint8)
for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            y0 = bd_px + r * sq_px
            x0 = bd_px + c * sq_px
            img[y0:y0 + sq_px, x0:x0 + sq_px] = 0

# Add a thin ruler tick line under the board so you can verify with calipers.
# Marks every 20 mm for 100 mm total (5 ticks). Falls inside the bottom border.
tick_y = img_h - bd_px // 2
tick_h = bd_px // 3
for i in range(6):
    x = bd_px + i * sq_px
    img[tick_y - tick_h:tick_y + tick_h, x - 2:x + 2] = 0

out_png = "/Users/v/Downloads/69conference/chessboard_9x6_20mm.png"
cv2.imwrite(out_png, img)
print(f"Wrote {out_png}")
print(f"  image size: {img_w} x {img_h} px  ({img_w/DPI*MM_PER_IN:.1f} x {img_h/DPI*MM_PER_IN:.1f} mm)")
print(f"  board:      {COLS} x {ROWS} squares  ->  {COLS-1} x {ROWS-1} INNER corners")
print(f"  square:     {SQUARE_MM} mm = {sq_px} px @ {DPI} DPI")
print(f"  paper:      Fits on US Letter (216x279 mm) or A4 (210x297 mm) in landscape.")
print("")
print("Print checklist:")
print("  1. Open in Preview -> Print")
print("  2. Scale = 100%   (NOT 'Fit to Page')")
print("  3. Paper size matches what's in the tray")
print("  4. Borderless = OFF")
print("  5. After printing, measure 5 squares across with a ruler -> must be 100.0 mm")
print("     The tick row under the board marks every 20 mm for easy verification.")
