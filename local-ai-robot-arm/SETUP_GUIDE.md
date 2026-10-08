# Setup Guide — Voice-Controlled Robot Arm

A step-by-step guide to building and running this system from scratch.

**No programming knowledge needed.** Every command is written out to copy and
paste. After each step there is a **✅ Check** telling you what success looks
like, so you never move on from a broken step.

Budget about **2 hours** the first time: 30 min hardware, 30 min software
install (mostly waiting for downloads), 30 min calibration, 30 min testing.

---

## Contents

1. [What you are building](#1-what-you-are-building)
2. [What you need](#2-what-you-need)
3. [Build the camera gantry](#3-build-the-camera-gantry)
4. [Connect the hardware](#4-connect-the-hardware)
5. [Install the software](#5-install-the-software)
6. [Calibrate](#6-calibrate)
7. [Run it](#7-run-it)
8. [Everyday use](#8-everyday-use)
9. [When something goes wrong](#9-when-something-goes-wrong)

---

## 1. What you are building

A robot arm that picks up coloured cubes when you **talk to it**. Everything
runs on the laptop — no internet, no cloud account, no API key.

```
   you speak  →  laptop hears   →  laptop understands  →  laptop sees
                  (Nemotron)         (Qwen3)               (OWLv2)
                                                               ↓
                                              robot arm moves and picks it up
```

You say *"pick the green cube"*; a camera above the desk finds it; the arm
reaches down, sucks it up with a vacuum cup, and holds it. Say *"put it in the
box"* and it does. Say *"wait"* and it stops and holds.

---

## 2. What you need

### Hardware

| Item | Notes |
|---|---|
| **myCobot 280 M5** robot arm | with its own power supply — do not power from USB |
| **Suction pump kit** | the standard Elephant Robotics vacuum pump |
| **USB webcam, 1080p** | any normal UVC webcam |
| **Apple Silicon Mac** | M1/M2/M3/M4, 16 GB memory recommended |
| **2 aluminium extrusions** + **1 straight connector** | for the camera gantry |
| **Base plate or clamp** | to hold the gantry upright |
| **Coloured foam cubes** | ~40 mm. Green and red work well |
| **Small cardboard box** | the "put it in the box" target |
| **Microphone** | a plug-in USB/3.5 mm mic is strongly preferred — see §4 |

### A note on cube size

The software assumes **40 mm** cubes. If yours are a different size, measure
one with a ruler and tell your installer — one number in the software has to
match (`CUBE_HEIGHT_MM`).

---

## 3. Build the camera gantry

The camera looks **straight down** at the desk from about **75 cm** up.

```
          ┌───────┐  ← camera, pointing DOWN
          │ •     │
    ══════╧═══════╡   ← horizontal extrusion (the arm of the gantry)
                  ║
                  ║   ← vertical extrusion
                  ║      joined by a STRAIGHT CONNECTOR
                  ║
    ══════════════╩═══  ← base plate / desk clamp
```

**Steps**

1. Join the two aluminium extrusions into an **L shape** with the straight
   connector — one vertical, one horizontal.
2. Fix the vertical piece to a base plate or clamp it to the desk edge. It must
   not wobble; any movement ruins the calibration.
3. Mount the camera at the end of the horizontal arm, **lens pointing straight
   down** at the work area.
4. Set the height so the **camera lens is ~75 cm above the desk surface**.
5. Position the arm of the gantry so the camera is roughly **over the middle of
   the robot's working area**, not over the robot's base.

**Important:** once calibrated, **do not move the gantry or the camera.** If
either gets bumped, you must redo the calibration in §6.

> **Why 75 cm?** Lower means the cubes look bigger and are easier to detect,
> but the camera sees less of the desk. 75 cm is a good balance. The robot can
> only reach about 28 cm from its own base, so there is no benefit to seeing a
> huge area.

✅ **Check:** the gantry does not wobble when you nudge the desk, and the
camera points straight down (not at an angle).

---

## 4. Connect the hardware

1. **Robot power** — plug in the robot's own power brick. Switch it on.
2. **Robot USB** — connect the robot to the Mac with its USB cable.
3. **Camera USB** — connect the webcam to the Mac.
4. **Pump** — connect the suction pump to the robot as per the Elephant
   Robotics instructions. The pump is driven by the robot, not the Mac.
5. **Microphone** — plug in the external mic.

### Please use an external microphone

The laptop's **built-in** microphone sits inside the laptop on the same desk as
the robot, so it picks up vibration from the pump. We measured this:

| Microphone | How much the running pump raises the noise |
|---|---|
| External mic | **1.8×** |
| Laptop built-in | **3.7×** |

With the built-in mic, the pump can be nearly as loud as your voice, and the
system stops hearing you while it is holding a cube. An external mic — ideally
a headset or one placed away from the robot — avoids this.

### Cable routing

Tape the robot's USB cable down so it has slack and cannot be tugged. During
calibration you move the arm by hand, and pulling the cable disconnects the
robot mid-calibration.

✅ **Check:** the robot's lights are on, and the arm is powered.

---

## 5. Install the software

Open the **Terminal** app on the Mac. Copy and paste each block, pressing
Enter after each, and wait for it to finish before the next.

**1. Go to the project folder**
```bash
cd ~/local-ai-robot-arm
```

**2. Create the Python environment** (one time only)
```bash
python3 -m venv mlx_env
```

**3. Install the software** (takes several minutes)
```bash
./mlx_env/bin/pip install -r requirements.txt
```

**4. Check it worked**
```bash
./mlx_env/bin/python test_system.py
```

You should see a list of ticks ending with something like `52/52 passed`.

> Some checks will fail until you have calibrated — that is expected at this
> stage. What matters is that the script **runs** and does not crash.

✅ **Check:** `test_system.py` runs and prints a results table.

### About the first run

The first time you start the main program it downloads three AI models (a few
GB). This happens **once** — after that everything runs offline. Leave it
connected to the internet for the first run only.

---

## 6. Calibrate

Calibration teaches the system **where things are**: it links what the camera
sees to where the robot must move. **Do this after any change to the camera,
the gantry, or the robot's position.**

### Step 6a — Check the camera

```bash
./mlx_env/bin/python test_camera.py
```

This checks the picture is sharp, bright, steady, and that your cube colours
are detectable.

✅ **Check:** mostly ticks. Pay attention to two lines:
- *"image is in focus"* — if it fails, adjust the camera's focus ring
- *"exposure is STABLE"* — if it fails, the lighting is changing; close blinds
  or turn on steady room lighting

### Step 6b — Teach it where the desk is

This is the important one. You show the system several points on the desk, and
for each you touch the exact same spot with the robot's pump tip.

**Using a cube** (easiest — no printing needed):

```bash
./mlx_env/bin/python affine2d_calibrate_cube.py
```

A camera window opens and the arm goes limp so you can move it by hand.

For each point — **do 6 to 8 points, spread across the whole working area**:

1. Place the cube somewhere on the desk. Move the robot arm **out of the
   camera's view**.
2. Press **SPACE** — a red cross marks the cube.
3. Now gently move the robot arm by hand so the **pump tip touches the centre
   of the top of the cube**.
4. Press **SPACE** again — the point is recorded.
5. Move the cube somewhere else and repeat.

Press **Q** when you have 6–8 points. Press **U** to undo if you make a
mistake.

**Alternative, using printed markers:** print `aruco_markers.pdf` at *Actual
Size (100%)*, scatter the markers on the desk, and run
`affine2d_calibrate_multi.py` instead. Same idea.

✅ **Check:** it prints `mean X.XX mm`. **Under 5 mm is good.** If it is
higher, your touches were not accurate — run it again and take more care
putting the pump tip exactly on the cube centre.

### Step 6c — Check the height

```bash
./mlx_env/bin/python test_cube_hover.py
```

Press **1** for green (or **5** for red), wait for the green **STABLE** marker,
then press **SPACE**. The arm moves above the cube.

Use **`-`** and **`+`** to move the tip down and up 10 mm at a time, until the
pump tip just **touches the top of the cube**.

Note the **`tipZ=` number** shown at the top when it is touching — tell your
installer this number. It sets how deep the arm reaches when picking.

✅ **Check:** the pump tip sits directly over the cube centre, not off to one
side. If it is consistently off, redo step 6b.

---

## 7. Run it

```bash
./mlx_env/bin/python qwen_command.py
```

Wait for the models to load, then look for:

```
▶ mic is live — speak a command
Ready.
```

A camera window opens. At the bottom you will see a coloured dot:

| Dot | Meaning |
|---|---|
| ⚪ grey — *ready, speak* | it is listening for a command |
| 🟢 green — *HEARING YOU* | it can hear you **right now** |
| 🟠 orange — *THINKING* | it is working out what you said — wait |

**Say one command, then stop and wait.** Do not repeat yourself — repeating
makes it *worse*, because it joins your repeats into one long jumbled sentence.

### Commands it understands

| Say | It does |
|---|---|
| "pick the green cube" | picks up the green cube |
| "pick the red cube" | picks up the red cube |
| "place it in the box" | puts what it is holding into the box |
| "can you give it to me" | brings it to your hand |
| "wait" or "stop" | stops and holds the cube, waiting for your next order |
| "reset" or "go home" | returns to the starting position |
| "quit" | shuts down |

### Tips that make a real difference

- **Hold your hand closer to the robot** when asking for delivery — within
  about 20 cm of the robot's base. Further than that is out of its reach.
- **Say "reset", not "home"**, and **"cancel"** rather than "wait" if it is not
  hearing you. One-syllable words are harder for speech recognition.
- **Speak normally.** Shouting does not help; a steady clear voice does.

---

## 8. Everyday use

Once set up, starting the system is just:

```bash
cd ~/local-ai-robot-arm
./mlx_env/bin/python qwen_command.py
```

Press **Ctrl + C** in the Terminal to stop it.

**You do not need to recalibrate each day** — only if the camera, gantry, or
robot has been moved or bumped.

### Typing instead of speaking

If the room is too noisy (a trade show, for example), the system can be driven
by typing instead. Ask your installer to set `INPUT_MODE = "text"` — then you
type commands instead of speaking them, and everything else works the same.
This is much more reliable in loud environments.

---

## 9. When something goes wrong

### It does not hear me

1. Look at the dot in the camera window. Grey means it is not detecting your
   voice at all; green means it hears you and the problem is elsewhere.
2. Check the correct microphone is selected in **System Settings → Sound →
   Input**.
3. Check the microphone's **input volume** in the same place. Too high is as
   bad as too low.
4. If it only fails **while the arm is holding a cube**, it is the pump noise —
   use an external mic (§4), or put something soft under the pump to quieten it.

### It hears me but does the wrong thing

Speech recognition occasionally mishears. If it keeps mishearing one phrase,
try a different wording — "put it in the box" instead of "place it in the box".

### The arm does not reach / says "I can't reach that"

The robot can only reach about **28 cm** from its base. Move the cube, the box,
or your hand closer to the robot.

### It misses the cube

Redo the calibration (§6b). Any bump to the camera or gantry breaks it.

### The arm picks up the cube but drops it

The pump is not sealing. Usually the arm is not reaching quite deep enough —
redo §6c and tell your installer the `tipZ` number.

### Nothing works / I want to check everything

```bash
./mlx_env/bin/python test_system.py --all
```

This checks the configuration, the calibration, the robot connection, and the
camera, then prints what is wrong.

---

## Getting help

If you report a problem, these files are written automatically while the system
runs and contain everything needed to diagnose it:

| File | Contains |
|---|---|
| `session.log` | everything printed during the session |
| `voice.csv` | what the microphone heard, and when |
| `voice_clips/` | audio recordings of each command |
| `timing.csv` | how long each movement took |

Send those along with a description of what you expected and what happened.
