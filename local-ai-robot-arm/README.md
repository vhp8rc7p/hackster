# Voice-Controlled Pick-and-Place Robot Arm

A MacBook-Air-powered pick-and-place demo: a **mycobot 280** with suction
pump picks colored cubes (and credit cards / hands) under verbal command.
All three AI models — speech-to-text, the language model that parses
commands, and the open-vocabulary object detector — run **locally on the
Mac**, no cloud APIs.

## What it does

```
voice (mic)
  └─► STT: Nemotron 3.5 streaming  (MLX, on Mac)
       └─► text command
            └─► LLM: Qwen3-1.7B  (MLX, on Mac)
                 └─► JSON plan {action, object, target}
                      └─► OWLv2  (PyTorch + MPS, on Mac)
                           └─► pixel location of the requested object
                                └─► back-project to 3D via T_cam2base
                                     └─► ikpy → joint angles
                                          └─► robot moves + suction pump
```

Example commands:

```
> pick the green cube
> place it in the cardboard box
> bring the pink cube to my hand
> home
> stop
```

## Hardware

- **Robot:** Elephant Robotics myCobot 280 M5 + suction-pump end-effector
- **Camera:** USB webcam, 1920×1080, mounted on a gantry **above** the workspace
  (eye-to-hand configuration — camera is fixed, looks down)
- **Mac:** Apple Silicon (M1/M2/M3 — M2 Air is what this was built on)
- **Mic:** built-in or external (script pins to the built-in mic by name)

## Software setup

1. Install Python 3.11+ and create a venv:
   ```bash
   python3 -m venv mlx_env
   source mlx_env/bin/activate
   pip install -r requirements.txt
   ```
2. Plug in the robot. Find its serial port:
   ```bash
   ls /dev/tty.usbserial-*
   ```
   Edit `SERIAL_PORT` at the top of `qwen_command.py` and `handeye_calibrate.py`.
3. Plug in the USB camera. Confirm it shows up as `CAMERA_ID = 0`. If you
   have multiple cameras, increment until you see the gantry view.

## Calibration (do this once per camera mount)

> **Setting this up for the first time, or handing it to someone else?**
> Read **[SETUP_GUIDE.md](SETUP_GUIDE.md)** instead — it walks through the
> hardware build, install, and calibration step by step, with no assumed
> programming knowledge.

The pipeline runs on a **2D affine** calibration (`CALIB_MODE = "affine"`):
a direct pixel → robot-XY mapping of the work surface. That is the only
calibration the demo needs.

Pick **either** method — both write `calibration_affine2d.json`:

### Option A — with a cube (no printing)

```bash
./mlx_env/bin/python affine2d_calibrate_cube.py
```

Move one coloured cube to 6–8 spots. At each: SPACE to lock the cube's pixel,
touch the cube top with the pump tip, SPACE again. Q to solve.

Because it is calibrated at **cube-top height**, cube picks absorb the parallax
that a table-plane fit leaves behind — slightly better for cubes, slightly
worse for flat objects.

### Option B — with ArUco markers

```bash
./mlx_env/bin/python make_aruco_markers.py     # generates aruco_markers.pdf
./mlx_env/bin/python affine2d_calibrate_multi.py
```

Print at **Actual Size (100%)**, scatter the markers, then touch each one's
centre with the pump tip in the live window.

**Target: mean residual < 5 mm.** Both scripts print it when they solve.

### Then check the height

```bash
./mlx_env/bin/python test_cube_hover.py
```

SPACE to hover over a cube, `-`/`+` to adjust 10 mm at a time until the tip
touches the cube top. That `tipZ` sets `TOUCH_ABOVE`.

### Camera intrinsics (rarely needed)

`gantry_calib/intrinsics.json` holds a 0.68 px calibration. **Affine mode does
not use it** for pick coordinates — only for masking the arm out of detections.
Redo with `calibrate_intrinsics_charuco.py` only if you change camera or lens.

### Legacy: 3D hand-eye

`handeye_calibrate.py` (chessboard on the end-effector) feeds the older
`CALIB_MODE = "handeye"` path. Not used by the current setup.

## Running the demo

```bash
python qwen_command.py
```

You'll see camera previews, model loading messages, and a prompt. You can
type or speak commands. Speech is detected automatically (VAD on RMS
energy). The first run downloads the three models — a few GB total — and
then everything runs offline.

Commands the LLM understands:

| Intent             | Example                                  |
|--------------------|------------------------------------------|
| pick               | `pick the green cube`                    |
| place              | `place it in the cardboard box`          |
| pick and place     | `put the pink cube on my hand`           |
| go home            | `home` / `reset`                         |
| stop / abort       | `stop` / `cancel` / `wait`               |
| quit               | `quit`                                   |

## Files

| File | Purpose |
|---|---|
| `qwen_command.py` | Main script: STT → LLM → OWL → IK → robot |
| `handeye_calibrate.py` | Pattern-based eye-to-hand calibration |
| `make_chessboard.py` | Generates the printable chessboard PNG |
| `chessboard_9x6_20mm.png` | The chessboard image (print at 100% scale) |
| `mycobot_280_m5.urdf` | Robot model used by ikpy for FK/IK |
| `calibration_result.json` | Current camera intrinsics + `T_cam2base` |
| `gantry_calib/intrinsics.json` | Standalone camera-intrinsics calibration |
| `tts/*.wav` | Pre-rendered speech prompts |
| `test_qwen_owl.py` | Dry-run: Qwen plan → OWL detection, no robot |
| `test_hand_detection.py` | Live OWL hand detection with confidence dump |
| `owl_live_preview.py` | Live preview of arbitrary OWL queries |
| `hand_follow_ik.py` | Continuous hand-tracking IK demo |
| `pick_cubes.py` | Earlier hard-coded pick demo (pre-LLM) |

## Tuning knobs

All in `qwen_command.py` near the top. The interesting ones:

| Constant | What it does |
|---|---|
| `HOVER_ABOVE` | mm above target before descending to pick |
| `TOUCH_ABOVE` | mm above target surface where pump engages (negative = press into target) |
| `OWL_THRESHOLD` | minimum detection confidence (default 0.08) |
| `OWL_MAX_BBOX_AREA_FRAC` | reject detections covering > N% of frame (kills "whole desk = box") |
| `STABLE_HAND_SECONDS` | how long the hand must hold still before delivery |
| `STABLE_BOX_SECONDS` | same, for a target container |
| `ROBOT_EXCLUSION_RADIUS_PX` | mask around projected pump tip to stop self-detection |

## Troubleshooting

- **Camera not found / wrong device:** unplug, replug, re-probe with
  `ls /dev/video*` or by trying CAMERA_ID 1/2/3.
- **Serial port errors:** the mycobot's USB serial id changes per cable
  and per arm. Re-check with `ls /dev/tty.usbserial-*`.
- **Picks miss by ~5–10 mm:** redo hand-eye calibration with more rotation
  diversity and a rigid pattern mount.
- **OWL detects the arm as a "hand":** raise `ROBOT_EXCLUSION_RADIUS_PX`.
- **STT hears Chinese / other languages:** the script forces `language="en-US"`.
  Background noise can still trip it; speak clearly.
- **"Wait" / "stop" not interrupting:** check `INTERRUPT_WORDS` in the
  script. Short words have special-case handling.
