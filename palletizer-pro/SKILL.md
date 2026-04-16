---
name: palletizer-pro
description: Control the myPalletizer 260 M5 robot arm through natural language. Use this skill whenever the user wants to move the robot arm, pick or place objects, detect cubes by color, scan for ArUco markers, run calibration, or do anything physical with the palletizer. Trigger even if the user just says "pick the red cube", "find marker 5", "calibrate", or "put it on ID 10" — any robotic arm task should use this skill.
---

# PalletizerPro — myPalletizer 260 M5 Controller

You are the interface between natural language and a 4-DOF robot arm. Your job is to understand what physical action the user wants, pick the right script, and run it with the right arguments.

## Hardware context

- **Arm**: myPalletizer 260 M5 (4-axis: X, Y, Z, Rz)
- **Serial port**: `COM4` at 115200 baud (Windows). If the user is on Linux/Mac, it may be `/dev/ttyUSB0` or `/dev/cu.usbserial-*` — ask if connection fails.
- **End-effector**: Suction pump (controlled via `pump.py`)
- **Camera**: USB camera at index 0 (640×480)
- **Calibration file**: `calibration_matrix.npy` — must exist before any vision-guided move

## Scripts and when to use them

All scripts live in the project root (same directory as `calibration_matrix.npy`). Run them with `python <script> [args]` from that directory.

### 1. Calibration — `hand_eye_cali_test.py`
**Use when**: user asks to calibrate, or when `calibration_matrix.npy` is missing/stale.

No arguments. Runs interactively:
1. User manually positions arm over ArUco marker, presses Enter
2. Script auto-samples a 3×3 grid of positions
3. Saves `calibration_matrix.npy`

> Calibration must complete before any vision-guided pick/place task. If a script fails with "找不到 calibration_matrix.npy", run this first.

---

### 2. Pick by pixel — `execute_pick_by_pixel.py`
**Use when**: user specifies exact pixel coordinates for the pick target.

```
python execute_pick_by_pixel.py --u <pixel_x> --v <pixel_y>
```

Example: "pick the object at pixel 320, 240" → `python execute_pick_by_pixel.py --u 320 --v 240`

---

### 3. Color-based scan + pick — `multi_color_pick_and_place.py`
**Use when**: user wants to detect and pick colored cubes (red or green) without knowing pixel coords.

```
python multi_color_pick_and_place.py
```

No arguments. Workflow:
1. Opens camera window showing detected cubes with labels
2. User presses `c` to confirm scan, `q` to quit
3. Script prints detected cubes (color + pixel coords)
4. **Stop here** — report findings to the user and ask which cube to pick
5. Use `execute_pick_by_pixel.py` with the chosen cube's coords to execute the pick

> Note: The script's `main()` is a stub — the detection/reporting is the useful part. Use `execute_pick_by_pixel.py` for the actual pick after the user chooses.

---

### 4. Pick ArUco marker by ID — `pick_aruco_by_id.py`
**Use when**: user wants to pick up an object tagged with a specific ArUco marker.

```
python pick_aruco_by_id.py --id <marker_id>
```

Example: "pick marker ID 5" → `python pick_aruco_by_id.py --id 5`

The script:
1. Moves arm to home/observe position
2. Scans camera for the target marker (averages 10 detections for stability)
3. Calculates world coordinates via calibration matrix
4. Executes pick → moves to `DROP_COORDS = [0, -150, 100]`
5. Returns home

---

### 5. Pick red cube → place on ArUco marker — `place_cube_on_aruco.py`
**Use when**: user wants to pick a red cube and place it on a specific ArUco marker.

```
python place_cube_on_aruco.py --id <target_marker_id>
```

Example: "put the red cube on marker 10" → `python place_cube_on_aruco.py --id 10`

The script:
1. Finds red cube by color
2. Finds target ArUco marker
3. Picks cube → places it on the marker
4. Returns home

---

## Coordinate system

| Parameter | Value | Meaning |
|-----------|-------|---------|
| SAFE_Z | 200 mm | Transit height (arm travels at this Z) |
| PICK_Z | 75 mm | Pickup height for cubes |
| PLACE_Z | 85 mm | Placement height (slightly higher to avoid collision) |
| DROP_COORDS | [0, -150, 100] | Default drop-off point |
| TOOL_OFFSET_X | -10 mm | Camera-to-gripper offset |
| TOOL_OFFSET_Y | -40 mm | Camera-to-gripper offset |
| DIR_X / DIR_Y | -1 | Pixel-to-robot axis flip |

## Handling user requests

**Route by intent:**

| User says | Script to run |
|-----------|--------------|
| "calibrate" / "run calibration" | `hand_eye_cali_test.py` |
| "pick at pixel X, Y" | `execute_pick_by_pixel.py --u X --v Y` |
| "find cubes" / "scan the table" / "what cubes are there?" | `multi_color_pick_and_place.py` (scan only) |
| "pick the red/green cube" | scan first, then `execute_pick_by_pixel.py` |
| "pick marker ID N" / "grab ID N" | `pick_aruco_by_id.py --id N` |
| "put red cube on marker N" / "place on ID N" | `place_cube_on_aruco.py --id N` |

**When something is ambiguous** (e.g., "pick the cube" without color or ID), ask one clarifying question before running anything.

**When a script fails:**
- "机器人连接失败" → check USB cable and COM port; ask user to verify port
- "找不到 calibration_matrix.npy" → run `hand_eye_cali_test.py` first
- "无法打开相机" → check camera is connected and not used by another app
- "无法稳定定位" / detection fail → improve lighting, check marker/cube is in frame

## Color tuning

The HSV ranges in the scripts may need tuning depending on lighting:
- Red: hue 0–10 and 170–180 (wraps around in HSV)
- Green: hue 35–77

If detection is unreliable, tell the user the HSV ranges can be adjusted in `multi_color_pick_and_place.py` under `COLOR_RANGES`.
