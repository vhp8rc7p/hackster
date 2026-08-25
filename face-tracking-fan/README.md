# Face-Tracking Fan

A myCobot 280 that keeps a fan pointed at your face, using only a laptop webcam.

The arm finds your face in 3D from a single camera, transforms that position into
its own coordinate frame, and continuously re-aims the end effector so the airflow
stays on you as you move around your desk.

No depth camera, no stereo rig, no markers.

## How it works

```
webcam frame -> face landmarks -> 3D point in CAMERA frame
             -> rigid transform -> 3D point in ROBOT BASE frame
             -> pointing IK     -> joint angles -> pymycobot
```

**Depth from one camera.** Adult interpupillary distance averages 63 mm and varies
little between people. MediaPipe's landmark model reports both iris centres, so the
pixel distance between them converts directly to range. This is far steadier than
using face bounding-box width, whose edges wobble with detector confidence.

**Pointing IK, not full 6-axis IK.** A fan is rotationally symmetric, so its roll
doesn't matter. That lets the problem decompose:

- J1 swings the arm's working plane onto the target
- J2 + J3 park the wrist at a fixed spot in that plane, clear of the desk
- J4 sets the elevation so the tool axis intersects the target
- J5 + J6 stay parked

The tool sits ahead of the wrist, so rotating the wrist also moves the tool. The
solver iterates three times to converge. Residual aiming error is 0.000 degrees,
verified against forward kinematics.

## Hardware

- Elephant Robotics myCobot 280 (M5) — 6-axis, 280 mm reach, 250 g payload
- Any laptop with a built-in webcam
- A small 5 V USB fan, mounted to the end effector

## Setup

```bash
pip install pymycobot opencv-python mediapipe numpy
```

Download the face landmark model into this directory:

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Without it the code falls back to an OpenCV Haar cascade, which is much worse —
about a 25% detection rate, and prone to locking onto door frames.

### Configure

Two things must be set in the config block at the top of the script.

**1. Serial port.** Find it with `python -m serial.tools.list_ports`, then set:

```python
PORT = "COM4"      # Windows; /dev/ttyUSB0 on Linux
```

**2. Camera position.** Tape-measure from the arm's base rotation axis to the
webcam lens, in metres, **from the robot's point of view**:

```python
CAM_POS_BASE = np.array([0.35, -0.45, 0.25])   # x forward, y left, z up
```

Note the sign convention — `+y` is the robot's left, not yours. If the arm sits to
your left, then you are on its right, and that value is negative. Getting this
backwards makes the arm track your face perfectly and aim in the wrong direction.

## Running

```bash
python face_track_mycobot280.py
```

Motion is **disabled at startup**. Press space to enable it.

| Key | Action |
|-----|--------|
| `space` | Arm / disarm motion |
| `t` | Test pose — point straight forward, level |
| `h` | Return to home |
| `a` / `d` | Trim aim left / right |
| `w` / `s` | Trim aim up / down |
| `p` | Print current trim values |
| `q` | Quit, leaving the arm holding its pose |

Start disarmed and press `t` first. The arm should point forward and level. If it
folds the wrong way, flip `PITCH_SIGN`.

Once tracking, use `a`/`d`/`w`/`s` to correct for how your laptop lid is tilted and
rotated, then `p` to print the values for permanent entry into the config.

## Calibration notes

`FX`/`FY` are set to 950 px, a rough figure for a ~68° horizontal FOV webcam at
1280x720. Replace with `cv2.calibrateCamera()` output for better accuracy.

Link dimensions are the myCobot 280's published geometry. `TOOL_LEN` is 121.8 mm
for a bare flange — **increase it by however far your fan sits past the wrist**,
since the aim depends on it.

## Known limits

- **J4 runs out of travel at ~35° of upward elevation.** Fine seated (~15°), but a
  standing user close to the arm will exceed it. The overlay flags this and holds
  position rather than straining. Raising `WRIST_Z` recovers range.
- **250 g payload.** Fan plus mount should be weighed. Mass at the end of an
  extended arm loads the joints hardest, and an overloaded arm sags and aims low.
- **Runs at ~13 Hz**, limited by landmark model inference.

## Safety

The arm moves autonomously, in response to a person, within reach of their face.

- Motion disabled until explicitly enabled
- Reads its actual pose at startup, so it doesn't lurch from an assumed one
- Joint speed capped at 60 deg/s
- 1.5° dead band, so servos don't buzz
- All commands clamped to rated joint limits
- Unreachable targets flagged, not attempted
- Holds still after 2 s without a face
- **Holds torque on exit** — it does not go limp and fall on the desk

## License

MIT
