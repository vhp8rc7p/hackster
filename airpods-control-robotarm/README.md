# AirPods → myCobot 280

Head-pose teleoperation of an Elephant Robotics myCobot 280, using the IMU in
AirPods Pro.

                                                  ┌─UDP :9870─▶ bridge.py ─serial─▶ myCobot 280
    AirPods IMU ──CoreMotion──▶ headtrack (Swift) ─┤                 10 Hz
                                   ~25 Hz          └─UDP :9871─▶ viz.py ──SSE──▶ 3D head in browser

Both consumers get every packet; either can run without the other.

## Setup

Needs macOS 14+ (`CMHeadphoneMotionManager`), Xcode command line tools, and
AirPods with motion sensors — Pro 1/2, AirPods 3/4, Max, or Beats Fit Pro.
AirPods 1/2 have no IMU and will not work.

    python3 -m venv .venv
    ./.venv/bin/pip install -r requirements.txt
    ./headtrack/build.sh

The build embeds `Info.plist` into the binary. That is not optional: without
`NSMotionUsageDescription` inside the executable, macOS never shows the motion
permission prompt and CoreMotion silently delivers nothing.

First run will ask for motion access. Grant it, or you get `auth=0` and no
samples.

## Run

Terminal 1 — start the head tracker (grant motion access when prompted):

    ./headtrack/headtrack

Terminal 2 — dry run first, no motion, just watch the numbers:

    ./.venv/bin/python bridge.py

When the mapping looks right, drive the arm for real:

    ./.venv/bin/python bridge.py --live --gain 0.8 --speed 30

Key controls need a real terminal. Piped or scripted runs disable them and stay
disengaged unless you pass `--auto-engage` (with `--duration N` to self-stop).

## Visualizer

    ./headtrack/headtrack          # terminal 1
    ./.venv/bin/python viz.py      # terminal 2 — opens http://127.0.0.1:8770/

A photogrammetry head scan mirroring your orientation in real time, with each
axis labelled by the joint it drives. Drag to orbit, scroll to zoom, "Recenter
yaw" to re-zero the heading. Stdlib only — no pip packages, no websockets.

Add `?pose=pitch,roll,yaw` to freeze a pose and skip the live stream — useful
for checking the view with no AirPods connected:

    http://127.0.0.1:8770/?pose=-12,0,22

### The model

`models/LeePerrySmith.glb` — 3D head scan by Lee Perry-Smith (Infinite
Realities), CC&nbsp;BY&nbsp;3.0, via the three.js examples. Colour and normal
maps alongside it. Attribution is required and is shown on the page.

It is a **bust**, not a bare head: it runs down to the shoulders, so its
bounding box is useless for placing anything. The ear position is measured off
the mesh instead — the ear spans y 0.5–2.0 in model units and its concha sits
at `|x|=1.85, y=1.25, z=-0.17`. The head is pivoted on that ear line rather
than the bbox centre, which is roughly where a real head turns.

## Controls

| key     | action                                                  |
|---------|---------------------------------------------------------|
| `SPACE` | engage / disengage the clutch                            |
| `c`     | recenter (recapture head + arm baselines)                |
| `r`     | release servos — arm goes limp                           |
| `q`     | quit                                                     |

## Mapping

| head motion     | joint              |
|-----------------|--------------------|
| turn  (yaw)     | J1 base rotation   |
| nod   (pitch)   | J2 shoulder        |
| tilt  (roll)    | J5 wrist pitch     |

J3, J4 and J6 hold at whatever they were when you engaged. Edit `AXIS_MAP` to
remap, or negate an entry's scale to flip that axis's direction.

## Safety

- **Nothing moves until you engage**, and engaging captures both baselines, so
  it never produces a jump.
- **Watchdog**: no head packet for 300 ms and the clutch drops automatically —
  covers AirPods disconnecting, going in the case, or `headtrack` dying.
- **Rate limit**: `MAX_STEP_DEG` 3°/tick at 10 Hz = 30°/s ceiling per joint, so
  a head jerk can't become an arm jerk.
- **Excursion cap**: `MAX_EXCURSION` 45° from the engage pose.
- **Soft joint limits**: `JOINT_LIMITS`, inside the mechanical range.

## Why it's built this way

**Yaw drifts.** AirPods have no magnetometer, so heading has no absolute
reference and creeps over minutes. Pitch and roll are gravity-referenced and
stay honest. Every engage recaptures the yaw baseline, which keeps drift
bounded to a single clutched session — the same trick VR controllers use.

**Joint space, not Cartesian.** `send_coords` runs IK on the arm's MCU: slower,
and it fails near singularities. `send_angles` is direct and predictable.

**10 Hz, not 25.** The serial link is the bottleneck. `set_fresh_mode(1)` makes
the arm act on the newest command instead of queueing a backlog.

## Verified

- AirPods Pro 2: CoreMotion authorized, ~25 Hz, pitch/roll/yaw live.
- Arm: responds at 115200, returns joint angles and coords.
- Pipeline: head motion drives J1/J2/J5 end to end in dry run.
- Not yet exercised: `--live`. Nothing has physically moved the arm.

## Hardware

Port `/dev/cu.usbserial-5AE20107941` @ 115200 (WCH CH9102, vendor `0x1a86`).
If it changes, `ls /dev/cu.usbserial-*` and edit `PORT` in `bridge.py`.

Needs AirPods with an IMU: Pro 1/2, AirPods 3/4, Max, or Beats Fit Pro.
AirPods 1/2 have no motion sensors and won't work.
