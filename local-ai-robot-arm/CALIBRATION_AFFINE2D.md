# 2D Affine Touch Calibration

Fast, accurate pixel-to-robot-base mapping for a **gantry-mounted camera**
looking down at a flat desk workspace (eye-to-hand setup). Uses ArUco
markers scattered across the workspace + physical touching with the pump
tip.

**Result:** a 2×3 affine matrix that maps camera pixels to robot base-frame
XY coordinates at the desk surface (z=0 plane).

## When to use this

- ✅ Camera fixed to a gantry above the workspace
- ✅ Objects to pick are all on the **flat desk surface** (cards, coins, paper)
- ✅ You want the simplest, most accurate calibration for this setup
- ❌ NOT for tall objects (cubes at height, hand held up) — those need 3D calibration
- ❌ NOT for cameras mounted on the arm (eye-in-hand)

Expected accuracy: **~1-3 mm mean residual** across the workspace.

---

## What you need

### Hardware
- **Robot** connected via USB serial (myCobot 280 M5 in this project)
- **Camera** rigidly mounted on the gantry, pointed straight down at the desk
- **Suction pump** attached to the arm's flange (or any tool of known length)
- **4-8 ArUco markers** (~25 mm each) printed from `DICT_6X6_50`, any IDs
- **Tape** to fix the markers flat to the desk

### Software prerequisites
- Camera **intrinsics** already calibrated → `gantry_calib/intrinsics.json`.
  If you replaced the camera or its focus, redo intrinsics first with
  `calibrate_intrinsics_charuco.py`.

---

## Step-by-step

### 1. Print + scatter markers

Print 4-8 ArUco markers from `DICT_6X6_50` at 25 mm side length. Any IDs
are fine — the script auto-detects whatever's visible.

**Tape them flat** across your workspace. Cover the area where you'll
actually be picking objects — not clustered in one spot. Corners of the
workspace, center, near/far from the robot base.

The more spread out, the better the affine fits across the workspace.

### 2. Verify the camera sees all markers

Open Preview and look at what the camera sees:

```bash
./mlx_env/bin/python -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
for _ in range(5): cap.read()
ok, f = cap.read()
cv2.imwrite('/tmp/cam.png', f)
cap.release()
" && open /tmp/cam.png
```

Confirm all markers are visible and reasonably in focus.

### 3. Run the calibration

```bash
./mlx_env/bin/python affine2d_calibrate_multi.py
```

**Phase A — lock marker pixel positions:**

1. Live camera window opens. Markers are outlined in green with their IDs.
2. When you see at least 4 markers all detected, **press SPACE**.
3. Script prints locked pixel positions and saves an annotated snapshot
   (`affine_markers.png`) — Preview opens it automatically so you can see
   which physical marker corresponds to which ID.

**Phase B — touch each marker's center with the pump tip:**

1. Terminal prompts you to connect the robot and home it.
2. Servos release; you can drag the arm by hand.
3. For each marker (in ID order), position the pump tip on the center of
   that marker, hold steady, press ENTER. Type `s` and ENTER to skip a
   marker you can't reach.
4. Script logs each touched position in base frame.

**Phase C — solve:**

Script fits a 2×3 affine using least-squares (each point weighted equally).
Reports per-marker residuals + mean + max. Writes to
`calibration_affine2d.json`.

Target: **mean residual < 3 mm**, **max < 5 mm**. Higher = touch precision
was poor, or a marker moved during the procedure.

### 4. Verify with the hover test

```bash
./mlx_env/bin/python test_affine_landing.py
```

Follow the on-screen prompts to hover above a marker. Visually check that
the pump lands within a few mm of marker center. Move the marker to
different spots and repeat.

---

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| Arm won't move when homing | Free-mode from previous crashed run OR joint past software limit | Physically move joints within ±160°; script now resets free mode automatically |
| `error 5` reported | Joint out of software limit | Drag the offending joint back into range |
| Residual > 5 mm | Poor touch precision or marker moved | Re-tape markers flat; touch same point on cup each time; gentle pressure |
| Three markers residual 0.000 + one at 14 mm | LMEDS outlier rejection with too few markers | Fixed in current script — uses least-squares now |
| Hover lands 25+ mm off marker at edges | Camera intrinsics wrong for current lens | Redo intrinsics with `calibrate_intrinsics_charuco.py` |

## Files this workflow creates / uses

| Path | Role |
|---|---|
| `calibrate_intrinsics_charuco.py` | (Prereq) camera intrinsics from a ChArUco board |
| `gantry_calib/intrinsics.json` | Camera K + distortion, loaded during calibration |
| `affine2d_calibrate_multi.py` | The main calibration script |
| `affine2d_calibrate.py` | Older single-marker version (kept for reference) |
| `affine_markers.png` | Snapshot showing which physical marker has which ID |
| **`calibration_affine2d.json`** | **Output: the 2×3 affine + residuals** |
| `test_affine_landing.py` | Hover test to verify accuracy |

## Notes on gantry height

Rough rule: calibration error scales linearly with camera-to-desk distance.
Shorter gantry = better accuracy. Tested residuals in this project:

- 70 cm gantry, 8 markers: ~3.1 mm mean
- 1 m gantry, 4 markers: ~1.0 mm mean (with least-squares fit — LMEDS gave
  misleading results with only 4 markers)

More markers helps compensate for greater height. 8 markers spread across
the workspace is a good default regardless of height.
