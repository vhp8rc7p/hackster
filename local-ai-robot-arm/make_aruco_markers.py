"""
Generate a PDF of ArUco markers for 2D affine touch calibration.

The affine calibration script (affine2d_calibrate_multi.py) expects markers
from DICT_6X6_50, physical size 25mm by default. This script produces one
Letter-sized PDF with N markers, cut lines, and IDs labeled.

Print with "Actual Size" (100%) — do NOT use "Fit to Page".

Edit the constants at the top to change:
  - MARKER_MM    physical side length (must match affine calib script)
  - IDS          which marker IDs to include (any subset of 0-49)
  - COLS/ROWS    layout on the page

Output: /Users/v/local-ai-robot-arm/aruco_markers.pdf
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

# ── config ──────────────────────────────────────────────────────────
MARKER_MM = 25.0           # physical side length of each black square
QUIET_MM = 6.0             # white border around each marker (needed for detection)
DICT = cv2.aruco.DICT_6X6_50
FIRST_ID = 5               # markers are numbered from here upward
MAX_IDS = 50               # DICT_6X6_50 holds ids 0-49
GAP_MM = 10.0              # space between marker tiles (also the cut margin)
MARGIN_MM = 12.0           # page margin

# Page + resolution
PAGE_W_MM, PAGE_H_MM = 215.9, 279.4    # US Letter portrait
DPI = 300

# Layout is computed to FILL the page — the old fixed 4x2 grid used only the
# top third of the sheet and wasted most of the paper. More markers also means
# more calibration points, which makes the 2D affine fit better.
_cell_mm = MARKER_MM + 2 * QUIET_MM
_label_mm = 5.0            # room under each tile for its printed id
COLS = max(1, int((PAGE_W_MM - 2 * MARGIN_MM + GAP_MM) // (_cell_mm + GAP_MM)))
ROWS = max(1, int((PAGE_H_MM - 2 * MARGIN_MM + GAP_MM) //
                  (_cell_mm + _label_mm + GAP_MM)))
IDS = list(range(FIRST_ID, min(FIRST_ID + COLS * ROWS, MAX_IDS)))

OUT_PATH = "/Users/v/local-ai-robot-arm/aruco_markers.pdf"


def make_one(mid):
    """Render one ArUco marker as a PIL image with quiet border."""
    d = cv2.aruco.getPredefinedDictionary(DICT)
    px = int(round(MARKER_MM / 25.4 * DPI))
    q = int(round(QUIET_MM / 25.4 * DPI))
    img = cv2.aruco.generateImageMarker(d, mid, px)
    tot = px + 2 * q
    bordered = np.full((tot, tot), 255, np.uint8)
    bordered[q:q + px, q:q + px] = img
    return Image.fromarray(bordered), tot


def main():
    assert len(IDS) <= COLS * ROWS, f"COLS*ROWS={COLS*ROWS} < IDS={len(IDS)}"

    canvas_w = int(round(PAGE_W_MM / 25.4 * DPI))
    canvas_h = int(round(PAGE_H_MM / 25.4 * DPI))
    canvas = Image.new("L", (canvas_w, canvas_h), 255)

    _, cell = make_one(IDS[0])
    gap = int(round(GAP_MM / 25.4 * DPI))
    label = int(round(_label_mm / 25.4 * DPI))
    grid_w = COLS * cell + (COLS - 1) * gap
    grid_h = ROWS * (cell + label) + (ROWS - 1) * gap
    # centre the grid on the page so the margins are even top/bottom
    x0 = (canvas_w - grid_w) // 2
    y0 = max(int(round(MARGIN_MM / 25.4 * DPI)), (canvas_h - grid_h) // 2)

    draw = ImageDraw.Draw(canvas)
    for i, mid in enumerate(IDS):
        r, c = divmod(i, COLS)
        m, _ = make_one(mid)
        x = x0 + c * (cell + gap)
        y = y0 + r * (cell + label + gap)
        canvas.paste(m, (x, y))
        draw.text((x, y + cell + 4), f"id {mid}  {MARKER_MM:.0f}mm", fill=0)

    canvas.save(OUT_PATH, "PDF", resolution=DPI)
    print(f"Wrote {OUT_PATH}")
    print(f"  {len(IDS)} markers (ids {IDS}), {MARKER_MM:.0f} mm each, DICT_6X6_50")
    print(f"  Print: US Letter portrait, Scale=100% (Actual Size)")
    print(f"  Verify with ruler: each black square edge = {MARKER_MM:.0f} mm exactly")


if __name__ == "__main__":
    main()
