# Things used in this project

## Hardware components

| Qty | Component | Notes |
|---:|---|---|
| 1 | **Elephant Robotics myCobot 280 M5** | 6-DOF desktop robot arm, 280 mm reach, ~250 g payload. The M5Stack-based version with USB-serial control. |
| 1 | **myCobot Suction Pump kit (electric)** | The standard Elephant Robotics suction-cup end-effector. Plugs into the M5 IO pins; controlled over the same serial channel that drives the joints. |
| 1 | **Apple MacBook Air (M1/M2/M3)** | Runs all three AI models locally. Tested on M2 Air, 16 GB unified memory. Any Apple Silicon Mac with ≥ 8 GB will work. |
| 1 | **USB webcam, 1080p** | Any 1080p USB UVC webcam (Logitech C920 or similar). Mounted on a gantry **above** the workspace, looking straight down. This is an *eye-to-hand* setup — the camera is fixed, not on the arm. |
| 1 | **Gantry / camera mount** | Anything rigid that holds the camera ~50–70 cm above the workspace, pointing down. A microphone-arm desk clamp, a tripod over the desk, or a 2020-extrusion frame all work. |
| 1 | **USB-C hub or extra ports** | The MacBook Air needs to connect: robot serial, USB webcam, and optionally a charger. A 3-port hub keeps everything plugged in. |
| 1 | **Built-in MacBook microphone** | The script auto-pins to the MacBook's mic; external USB mics also work. (Avoid Bluetooth mics — sample rate negotiation can drop the audio thread.) |
| 4 | **Colored foam cubes**, ~25 mm | Green, blue, pink, yellow. Cheap craft-store cubes — solid colors help OWLv2 disambiguate. |
| 1 | **Small cardboard box** | Any container ~10×10×5 cm. Used as the "place destination." |
| 1 | **Printed chessboard pattern** (optional) | 9×6 inner corners, 15 mm squares — used for one of the calibration methods. PDF included in the repo. |
| 1 | **Printed ArUco marker** (recommended) | DICT_6X6_50, ID 3, ~100 mm side. Used for touch-based hand-eye calibration. PDF included. |
| 1 | **Power for the robot** | The myCobot 280 ships with its own DC power brick — keep using that, don't try to power it from the USB. |

## Software apps and online services

| Category | Name | Where it runs | Why |
|---|---|---|---|
| **LLM** | **Qwen3-1.7B** (via `mlx-lm`) | MacBook (MLX, Apple Silicon) | Parses natural language voice commands into structured JSON action plans. Small enough to load instantly, smart enough to extract `{action, object, target}`. |
| **Speech-to-Text** | **Nemotron 3.5 Streaming ASR (0.6B)** (via `mlx-audio`) | MacBook (MLX) | Real-time on-device transcription. English-only mode is forced to avoid the model drifting into other languages on background noise. Parakeet-TDT 0.6B is included as a toggleable alternative. |
| **Object Detection** | **OWLv2** (Google, via `transformers`) | MacBook (PyTorch, MPS backend) | Open-vocabulary detector — you ask for "a green cube" or "a credit card" by text, no retraining needed. This is the single most important component for making the demo flexible. |
| **Inverse Kinematics** | **ikpy** | MacBook (Python) | Solves joint angles for a target end-effector pose, with the constraint that the suction pump points straight down. The mycobot's built-in `send_coords` can't enforce orientation reliably; ikpy can. |
| **Robot control** | **pymycobot** | MacBook (Python) | Official Elephant Robotics Python library — sends joint angles, reads positions, toggles the pump valve pins. |
| **Computer vision** | **OpenCV (`opencv-python`)** | MacBook | Camera capture, ArUco detection, PnP, hand-eye calibration solvers, chessboard corner detection. |
| **Math / arrays** | **NumPy** | MacBook | Universal Python math library. Used everywhere. |
| **Audio I/O** | **sounddevice + soundfile** | MacBook | Microphone capture for the VAD/STT pipeline; playing back pre-recorded TTS prompts. |
| **Text-to-Speech** | **macOS `say`** *(or pre-rendered WAVs)* | MacBook | Built-in macOS TTS for spoken feedback ("Picking it up", "I can't reach that"). The repo ships with 11 pre-rendered WAV clips so playback is instant; switching to live `say` adds ~300 ms but speaks anything. |
| **3D model** | **mycobot_280_m5.urdf** | MacBook (loaded by ikpy) | The official Elephant Robotics URDF, describing the robot's kinematic chain. Used for FK/IK math. |

## Hand tools and fabrication machines

| Qty | Tool | Used for |
|---:|---|---|
| 1 | **Printer** (color or B/W) | Printing the chessboard and ArUco calibration patterns. **Set scale to 100% / Actual Size** — never "Fit to Page" — or the patterns will be off by 5–10 %. |
| 1 | **Metal ruler or calipers** | Verifying printed pattern sizes after printing. (Printers often shrink to ~93 % even at "100 %" — measuring lets the script know the true size.) |
| 1 | **Tape / glue stick / spray adhesive** | Mounting the printed chessboard to a piece of rigid cardboard. Paper flex during calibration is a major source of error; rigid backing is mandatory. |
| 1 | **Cardboard / foam board** | The chessboard backing. Any rigid flat surface ~A5 size. |
| 1 | **Screwdriver / Allen keys** | If you need to remove and re-attach the suction pump for any reason. |

---

### Notes for would-be builders

- **No 3D printing required.** No custom hardware. Everything plugs in via USB.
- **No cloud APIs.** All three AI models download once (~3 GB total) and then run offline forever.
- **No GPU other than the MacBook's GPU.** MLX uses Apple Silicon directly; PyTorch uses the MPS backend. A Mac with an M-series chip is all you need.
- **No ROS.** The whole stack is plain Python and the official `pymycobot` library.
- **Total parts cost (excluding the Mac):** myCobot 280 + pump kit ≈ $750; webcam ≈ $50; everything else is craft-store or paper.
