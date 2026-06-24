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

You need two calibrations: **camera intrinsics** (lens model) and
**eye-to-hand** (where the camera is relative to the robot base).

### 1. Camera intrinsics

Already done — `gantry_calib/intrinsics.json` contains a 0.58 px reprojection
error calibration. Only redo if you change the camera or lens.

### 2. Hand-eye (camera → robot base)

Print the chessboard pattern (`chessboard_9x6_20mm.png`) and **measure it**.
Printers usually don't print at exactly the intended size, so set
`SQUARE_MM` in `handeye_calibrate.py` to whatever you actually measure
(measure across 5 squares for accuracy).

Tape the printout to **rigid cardboard** and mount on the end-effector
(side of the pump tube works; exact mount point doesn't matter — the solver
figures it out).

Then:

```bash
python handeye_calibrate.py
```

- Servos release so you can drag the arm by hand
- Position the arm so the chessboard is in camera view, hold steady, press
  **SPACE** to capture
- Aim for **12–20 poses** with **rotation diversity on all 3 axes**
  (twist the wrist using J4/J5/J6 — don't just slide flat)
- Press **ENTER** to solve. The script tries 5 OpenCV hand-eye methods and
  keeps the best by residual
- Output overwrites `calibration_result.json` (the old one is backed up to
  `calibration_result.json.touch.bak`)

Target residual: **< 5 mm**. Higher means rotation diversity was poor,
the mount flexed, or the URDF doesn't match your particular arm.

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
