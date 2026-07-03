# Tic ArUco Toe

A 4-axis robot arm that plays tic-tac-toe against a human, using a webcam
to read the board and the minimax algorithm to pick the best move.

Companion project to the Hackster article
["Tic ArUco Toe: A 4-Axis Robot That Plays Tic-Tac-Toe (and Doesn't Lose)"](https://www.hackster.io/Elephant-Robotics-Official).

## Hardware

- Elephant Robotics **myPalletizer 260** (M5 or Pi version)
- Vacuum pump (comes with the palletizer pack)
- USB webcam (720p+), mounted top-down over the board
- Printed 3×3 grid board (A4 paper)
- 10 wooden game pieces with printed ArUco markers (DICT_4X4_50) on top
  - IDs 0–9. ID `10` is treated as X, others as O.

## Software

```bash
pip install -r requirements.txt
```

- Python 3.9+
- `pymycobot`
- `opencv-contrib-python` (must be `contrib`, standard `opencv-python`
  does not include ArUco)
- `numpy`
- `RPi.GPIO` (only on the Pi version)

## Files

| File | Purpose |
|---|---|
| `260startup.py` | pymycobot + GPIO pump init helpers |
| `pick_test.py` | Motion: hardcoded cell coords, pick-and-place sequence, pump control |
| `board_recognize.py` | Vision: ArUco calibration + real-time board state |

## Quick start

1. Wire up the arm, pump, and camera. Boot the Pi (or plug M5 into your PC).
2. Place 9 calibration markers on the printed board's cells.
3. Run `python board_recognize.py`, press `s` when all 9 are detected.
   Slot positions are locked in memory.
4. Clear the calibration markers. Start playing with the piece markers.
5. Run `python pick_test.py` to verify each of the 9 cell coords one at a time.
6. Adjust `TIC_TAC_TOE_COORDS` if any cell is off.

## How it works

- **Vision** — ArUco markers on both pieces and (during calibration) cells.
  OpenCV's `aruco.detectMarkers` gives sub-pixel centers and marker IDs in one call.
- **AI** — Vanilla minimax. Full game tree search since tic-tac-toe has only
  ~255k reachable states.
- **Motion** — 4-DOF `send_coords([x, y, z, yaw], speed)`. Hover → down →
  pump on → travel → down → pump off, all with generous sleeps.

Full walkthrough in the [Hackster article](https://www.hackster.io/Elephant-Robotics-Official).

## Gotchas

- **Marker size matters.** Print at least 15 mm square. Smaller markers
  flicker at typical webcam distances.
- **Hardcoded coordinates drift.** If the board moves >3 mm, pieces
  land on the cell borders. Re-measure or add an affine calibration.
- **Pump timing.** Wait 500 ms after `pump_off()` before moving,
  otherwise the piece skids.
- **Yaw traps.** On a 4-DOF palletizer, keep `yaw` constant (`10` in
  our coords) across all cells to avoid self-collision.
