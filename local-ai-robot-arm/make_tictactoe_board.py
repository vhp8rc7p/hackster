"""
Generate a 3x3 tic-tac-toe board as a printable PDF.

Default: 15cm x 15cm board, 5cm cells, white background with thin black
grid lines. Optionally adds 4 corner ArUco markers so a vision system can
auto-detect the board's position and orientation.

Print at ACTUAL SIZE (100%), not "Fit to Page".
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw

# ── config ──────────────────────────────────────────────────────────
CELL_MM = 50.0            # each cell side length
GRID_LINE_MM = 1.5        # thickness of grid lines
BORDER_MM = 3.0           # outer border thickness

# Corner ArUco markers for board detection. Set to 0 to skip.
CORNER_ARUCO_MM = 20.0    # side length of each corner marker (0 = no markers)
CORNER_MARKER_IDS = [40, 41, 42, 43]   # TL, TR, BR, BL
ARUCO_DICT = cv2.aruco.DICT_6X6_50

DPI = 300
PAGE_W_MM, PAGE_H_MM = 215.9, 279.4   # US Letter portrait
OUT_PATH = "/Users/v/local-ai-robot-arm/tictactoe_board.pdf"


def mm2px(mm):
    return int(round(mm / 25.4 * DPI))


def make_aruco(mid, size_mm):
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    px = mm2px(size_mm)
    img = cv2.aruco.generateImageMarker(d, mid, px)
    return Image.fromarray(img)


def main():
    board_side_mm = 3 * CELL_MM
    canvas_w = mm2px(PAGE_W_MM)
    canvas_h = mm2px(PAGE_H_MM)
    canvas = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(canvas)

    # Top-left of board on page (centered horizontally, ~30mm from top)
    if CORNER_ARUCO_MM > 0:
        # Leave room for corner markers to sit outside the grid
        top_margin_mm = 30 + CORNER_ARUCO_MM + 4
    else:
        top_margin_mm = 30
    board_x0 = (canvas_w - mm2px(board_side_mm)) // 2
    board_y0 = mm2px(top_margin_mm)
    board_x1 = board_x0 + mm2px(board_side_mm)
    board_y1 = board_y0 + mm2px(board_side_mm)

    # Outer border
    b = mm2px(BORDER_MM)
    draw.rectangle([board_x0, board_y0, board_x1, board_y1],
                   outline=0, width=b)

    # Interior grid lines (2 vertical + 2 horizontal)
    line_w = max(1, mm2px(GRID_LINE_MM))
    for i in (1, 2):
        x = board_x0 + i * mm2px(CELL_MM)
        draw.line([(x, board_y0), (x, board_y1)], fill=0, width=line_w)
        y = board_y0 + i * mm2px(CELL_MM)
        draw.line([(board_x0, y), (board_x1, y)], fill=0, width=line_w)

    # Optional corner ArUco markers
    if CORNER_ARUCO_MM > 0:
        gap = mm2px(3)   # 3mm gap between marker and board edge
        m_px = mm2px(CORNER_ARUCO_MM)
        # positions: (id_index, x, y) — TL, TR, BR, BL
        positions = [
            (0, board_x0 - m_px - gap, board_y0 - m_px - gap),                  # TL
            (1, board_x1 + gap,        board_y0 - m_px - gap),                  # TR
            (2, board_x1 + gap,        board_y1 + gap),                         # BR
            (3, board_x0 - m_px - gap, board_y1 + gap),                         # BL
        ]
        for idx, x, y in positions:
            mid = CORNER_MARKER_IDS[idx]
            marker = make_aruco(mid, CORNER_ARUCO_MM)
            canvas.paste(marker, (x, y))
            draw.text((x, y + m_px + 4), f"id {mid}", fill=0)

    # Print info footer
    footer_y = board_y1 + mm2px(20)
    txt = (f"Tic-Tac-Toe board  |  {board_side_mm:.0f}mm x {board_side_mm:.0f}mm  |  "
           f"cell {CELL_MM:.0f}mm  |  print at 100% Actual Size")
    draw.text((board_x0, footer_y), txt, fill=0)

    canvas.save(OUT_PATH, "PDF", resolution=DPI)
    print(f"Wrote {OUT_PATH}")
    print(f"  board: {board_side_mm:.0f} x {board_side_mm:.0f} mm  ({CELL_MM:.0f}mm cells)")
    if CORNER_ARUCO_MM > 0:
        print(f"  corner markers: {CORNER_ARUCO_MM:.0f}mm each, ids {CORNER_MARKER_IDS}")
    print(f"  Print: US Letter, Actual Size (100%)")


if __name__ == "__main__":
    main()
