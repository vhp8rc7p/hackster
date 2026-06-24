"""
Text → Qwen → action plan → OWL detect → IK + send_angles → robot moves.

Type natural language commands; Qwen extracts {object, action}; we run it.
Examples:
  > go to the pink cube
  > touch the yellow cube
  > home
  > quit
"""

import json
import re
import select
import sys
import time
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from mlx_lm import load as load_llm, generate as generate_llm
from mlx_audio.stt import load as load_stt
from pymycobot.mycobot280 import MyCobot280
from ikpy.chain import Chain
import sounddevice as sd
import soundfile as sf
import threading
import queue
import tempfile
import os

# ── config ─────────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/tty.usbserial-54780106801"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
CALIB_PATH = "calibration_result.json"
URDF_PATH = "mycobot_280_m5.urdf"

PUMP_LENGTH = 70.0   # was 50 — your real pump is ~7 cm
TABLE_Z_BASE_MM = 0.0
CUBE_HEIGHT_MM = 25.0
HAND_Z_BASE_MM = 100.0   # palm held about 10 cm above the desk   # how high above the desk you hold your hand for delivery
HOVER_ABOVE = 150.0
HAND_HOVER_ABOVE = 50.0   # smaller because the user catches the cube; keeps TCP in reach
TOUCH_ABOVE = -10.0  # pump presses 10mm into the cube top to absorb IK error at edge of reach
SPEED = 30
IK_ERR_LIMIT_MM = 15
MAX_JOINT_STEP_DEG = 180   # allow IK wrist-flips for picks at edge of reach
HOME_ANGLES = [0, 0, 0, 0, 0, 0]

# Suction pump (from pick_aruco.py / ai_pick.py)
PUMP_PIN = 2
VALVE_PIN = 5
PUMP_DWELL_S = 1.2     # time at touch height with pump on before lifting


_pump_engaged = False  # tracks whether the pump is currently holding something


def pump_on(mc):
    global _pump_engaged
    mc.set_basic_output(VALVE_PIN, 0)
    mc.set_basic_output(PUMP_PIN, 0)
    _pump_engaged = True


def pump_off(mc):
    global _pump_engaged
    mc.set_basic_output(PUMP_PIN, 1)
    mc.set_basic_output(VALVE_PIN, 1)
    _pump_engaged = False


# ── speech (TTS playback) ─────────────────────────────────────────────
import subprocess
TTS_DIR = "tts"


# ── system stats overlay ──────────────────────────────────────────────
try:
    import psutil
    _psutil_ok = True
    psutil.cpu_percent(interval=None)  # prime the rolling counter
except Exception:
    _psutil_ok = False

_stats = {"cpu": 0.0, "mem_pct": 0.0, "mem_gb": 0.0, "fps": 0.0, "owl_ms": 0.0}


# Per-inference metric log. Read this back later to see real numbers.
METRICS_LOG = "owl_metrics.csv"
_metrics_file = None


def _open_metrics_log():
    global _metrics_file
    try:
        _metrics_file = open(METRICS_LOG, "w", buffering=1)  # line-buffered
        _metrics_file.write("timestamp,site,owl_ms,cpu_pct,mem_pct,mem_gb,query\n")
        print(f"  logging OWL metrics → {METRICS_LOG}")
    except Exception as e:
        print(f"  could not open metrics log: {e}")


def _log_owl(site, ms, query=""):
    if _metrics_file is None:
        return
    try:
        _poll_sys_stats()
        line = (f"{time.time():.3f},{site},{ms:.1f},"
                f"{_stats['cpu']:.1f},{_stats['mem_pct']:.1f},"
                f"{_stats['mem_gb']:.2f},{query!r}\n")
        _metrics_file.write(line)
    except Exception:
        pass


# Dedicated debug log for tracing why detections do or don't get used.
DEBUG_LOG = "debug_trace.log"
_debug_file = None


def dlog(msg):
    """Append a timestamped line to debug_trace.log for post-mortem analysis."""
    global _debug_file
    if _debug_file is None:
        try:
            _debug_file = open(DEBUG_LOG, "w", buffering=1)
            _debug_file.write(f"=== session started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except Exception:
            return
    try:
        _debug_file.write(f"{time.time():.3f}  {msg}\n")
    except Exception:
        pass
_fps_t0 = time.time()
_fps_n = 0


def _update_fps():
    """Call once per frame in the show_preview loop."""
    global _fps_t0, _fps_n
    _fps_n += 1
    dt = time.time() - _fps_t0
    if dt >= 1.0:
        _stats["fps"] = _fps_n / dt
        _fps_n = 0
        _fps_t0 = time.time()


def _poll_sys_stats():
    if not _psutil_ok:
        return
    try:
        _stats["cpu"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        _stats["mem_pct"] = vm.percent
        _stats["mem_gb"] = vm.used / 1e9
    except Exception:
        pass


def _draw_stats_overlay(disp):
    _poll_sys_stats()
    h, w = disp.shape[:2]
    txt = (f"CPU {_stats['cpu']:.0f}%   "
           f"MEM {_stats['mem_pct']:.0f}% ({_stats['mem_gb']:.1f}G)   "
           f"FPS {_stats['fps']:.1f}   "
           f"OWL {_stats['owl_ms']:.0f}ms")
    # background bar at the top
    bar_h = 32
    cv2.rectangle(disp, (0, 0), (w, bar_h), (40, 40, 40), -1)
    cv2.putText(disp, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

SAY_TEXT = {
    "ok":              "Okay.",
    "picking":         "Picking it up.",
    "got_it":          "Got it.",
    "placing":         "Placing it now.",
    "released":        "Done.",
    "hold_still":      "Hold still, please.",
    "not_found":       "I can't find it.",
    "out_of_reach":    "I can't reach that.",
    "stopping":        "Stopping.",
    "going_home":      "Going home.",
    "didnt_understand":"I didn't understand.",
}


SPEECH_ENABLED = False   # set True to play TTS clips


def say(key):
    """Print a marker; play the audio file from TTS_DIR if SPEECH_ENABLED."""
    txt = SAY_TEXT.get(key, key)
    print(f"  🔊 {txt}")
    if not SPEECH_ENABLED:
        return
    for ext in (".mp3", ".wav", ".m4a"):
        path = os.path.join(TTS_DIR, key + ext)
        if os.path.exists(path):
            try:
                subprocess.Popen(["afplay", path],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except Exception:
                pass
            break


# ── voice listener ────────────────────────────────────────────────────
# Background thread continuously listens; emits transcribed utterances.
SAMPLE_RATE = 16000
VAD_FRAME_S = 0.05
VAD_RMS_THRESHOLD = 0.04     # raised again — background noise was still tripping it
VAD_SILENCE_END_S = 0.7      # trailing silence to end an utterance
VAD_MIN_UTTERANCE_S = 0.25
VAD_MAX_UTTERANCE_S = 8.0

voice_queue = queue.Queue()
_voice_stop = threading.Event()


def _find_macbook_mic():
    """Find the built-in MacBook mic so we don't get stuck on AirPods/USB hubs
    that don't support our 16kHz pipeline."""
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) <= 0:
                continue
            name = d.get("name", "")
            if "MacBook" in name or "macbook" in name or "麦克风" in name:
                return i, name
    except Exception:
        pass
    return None, None


def _voice_listener(stt_model):
    chunk_samples = int(SAMPLE_RATE * VAD_FRAME_S)
    silence_chunks_end = int(VAD_SILENCE_END_S / VAD_FRAME_S)
    max_chunks = int(VAD_MAX_UTTERANCE_S / VAD_FRAME_S)
    buffer, silence, speaking = [], 0, False
    last_heartbeat = time.time()
    device_idx, device_name = _find_macbook_mic()
    if device_idx is not None:
        print(f"  voice input: [{device_idx}] {device_name}")
    else:
        print(f"  voice input: system default (MacBook mic not found)")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=chunk_samples, device=device_idx) as stream:
            while not _voice_stop.is_set():
                data, _ = stream.read(chunk_samples)
                chunk = data.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                # Heartbeat every 30s so we know the listener is still alive
                if time.time() - last_heartbeat > 30:
                    last_heartbeat = time.time()
                    dlog(f"voice_listener heartbeat: rms={rms:.4f}")

                if rms > VAD_RMS_THRESHOLD:
                    if not speaking:
                        speaking = True
                    silence = 0
                    buffer.append(chunk)
                elif speaking:
                    buffer.append(chunk)
                    silence += 1
                    if silence >= silence_chunks_end or len(buffer) >= max_chunks:
                        audio = np.concatenate(buffer)
                        dur = len(audio) / SAMPLE_RATE
                        buffer, silence, speaking = [], 0, False
                        if dur < VAD_MIN_UTTERANCE_S:
                            continue
                        try:
                            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                            sf.write(tmp.name, audio, SAMPLE_RATE)
                            tmp.close()
                            result = stt_model.generate(tmp.name, language="en-US")
                            text = getattr(result, "text", "") or ""
                            text = text.strip()
                            os.unlink(tmp.name)
                            if not text:
                                continue
                            # Filter STT hallucinations on near-silence.
                            stripped = text.strip(".,!? ").lower()
                            FILLER = {"uh", "um", "ah", "eh", "oh", "hmm", "mm",
                                      "huh", "you", "yeah", "okay", "ok",
                                      "thank you", "thanks", "bye"}
                            words = stripped.split()
                            # ALWAYS let interrupt and release words through,
                            # even if short — they're critical for mid-action control.
                            is_critical = (
                                stripped in INTERRUPT_WORDS
                                or stripped in RELEASE_WORDS
                                or any(w in INTERRUPT_WORDS or w in RELEASE_WORDS
                                       for w in _strip_words(text))
                            )
                            if not is_critical:
                                if stripped in FILLER:
                                    print(f"  (ignored filler: {text!r})")
                                    continue
                                if len(words) < 2 and len(stripped) < 5:
                                    print(f"  (ignored too-short utterance: {text!r})")
                                    continue
                            print(f"\n  🎙️ heard: {text!r}")
                            voice_queue.put(text)
                        except Exception as e:
                            print(f"  ⚠ STT error: {e}")
    except Exception as e:
        print(f"  ⚠ voice listener died: {e}")


def start_voice(stt_model):
    t = threading.Thread(target=_voice_listener, args=(stt_model,), daemon=True)
    t.start()
    return t


# ── interrupt support ─────────────────────────────────────────────────
INTERRUPT_WORDS = {"stop", "s", "halt", "abort", "cancel", "wait"}
RELEASE_WORDS = {"drop", "release", "let", "go", "now"}


def _strip_words(text):
    """Lowercase + strip punctuation from each word."""
    return [w.strip(".,!?;:\"'`") for w in text.lower().split()]


def _is_interrupt_text(text):
    return any(w in INTERRUPT_WORDS for w in _strip_words(text))


def _is_release_text(text):
    return any(w in RELEASE_WORDS for w in _strip_words(text))


def _poll_stdin_nonblocking():
    """Return any line typed since last call, or None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            return sys.stdin.readline().rstrip()
        except Exception:
            return None
    return None


_release_flag = False  # set by check_interrupt when user says "drop"/"release"


def check_interrupt():
    """True if the user typed OR spoke a stop word. Also sets _release_flag
    if a release word was spoken (consumed by check_release_request)."""
    global _release_flag
    # typed
    line = _poll_stdin_nonblocking()
    if line is not None:
        t = line.strip().lower()
        if t in INTERRUPT_WORDS: return True
        if _is_release_text(t): _release_flag = True
    # spoken
    while True:
        try:
            spoken = voice_queue.get_nowait()
        except queue.Empty:
            break
        if _is_interrupt_text(spoken):
            return True
        if _is_release_text(spoken):
            _release_flag = True
    return False


def check_release_request():
    """True (once) if the user said/typed a release word since last check."""
    global _release_flag
    if _release_flag:
        _release_flag = False
        return True
    return False


_live_cap = None   # set in main; lets interruptible_sleep refresh the preview


def pump_preview(seconds=0.3, label=None):
    """Briefly refresh the camera preview window without doing OWL inference.
    Used between OWL inferences so the window doesn't freeze on idle frames."""
    if _live_cap is None:
        return
    end = time.time() + seconds
    while time.time() < end:
        try:
            ret, frame = _live_cap.read()
            if ret:
                disp = frame.copy()
                if label:
                    cv2.putText(disp, label, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                _draw_stats_overlay(disp)
                _update_fps()
                cv2.imshow("robot_view", disp)
        except Exception:
            pass
        cv2.waitKey(1)
        time.sleep(0.03)


def interruptible_sleep(seconds, mc=None):
    """Sleep that aborts on interrupt AND keeps the preview window alive.
    Returns True if interrupted."""
    end = time.time() + seconds
    last_frame_t = 0.0
    while time.time() < end:
        if check_interrupt():
            if mc is not None:
                try: mc.stop()
                except Exception: pass
            return True
        # Refresh preview every ~100ms so the window doesn't freeze.
        now = time.time()
        if _live_cap is not None and (now - last_frame_t) > 0.1:
            last_frame_t = now
            try:
                ret, frame = _live_cap.read()
                if ret:
                    disp = frame.copy()
                    cv2.putText(disp, "moving...", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
                    _draw_stats_overlay(disp)
                    _update_fps()
                    cv2.imshow("robot_view", disp)
                    cv2.waitKey(1)
            except Exception:
                pass
        else:
            time.sleep(0.03)
    return False

OWL_THRESHOLD = 0.08    # raised — 0.03 was accepting random green-ish noise
def _owl_threshold_for(query):
    """Per-query threshold hook. Currently uniform; kept as a function so we
    can re-introduce per-query thresholds later without touching call sites."""
    return OWL_THRESHOLD
# Reject detections whose bbox covers more than this fraction of the frame.
# Stops OWL from returning the entire desk as a "cardboard box".
OWL_MAX_BBOX_AREA_FRAC = 0.10   # tighter — was letting in 17%-area "desk+box" detections


# ── IK chain ──────────────────────────────────────────────────────────
chain = Chain.from_urdf_file(
    URDF_PATH, base_elements=['g_base'], last_link_vector=[0, 0, 0],
    active_links_mask=[False, False, True, True, True, True, True, True, False],
)


def solve_ik(target_xyz_mm, current_angles_deg, pointing_down=True):
    target_T = np.eye(4)
    target_T[:3, 3] = np.array(target_xyz_mm) / 1000.0
    init = [0.0] * len(chain.links)
    for i, deg in enumerate(current_angles_deg):
        init[i + 2] = float(np.radians(deg))
    if pointing_down:
        target_T[:3, :3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        j = chain.inverse_kinematics_frame(target_T, initial_position=init,
                                           orientation_mode="Z")
    else:
        j = chain.inverse_kinematics_frame(target_T, initial_position=init,
                                           orientation_mode=None)
    return [float(np.degrees(j[i + 2])) for i in range(6)]


def solve_ik_down(target_xyz_mm, current_angles_deg):
    """Backwards-compat alias."""
    return solve_ik(target_xyz_mm, current_angles_deg, pointing_down=True)


def fk_pos_mm(angles_deg):
    pose = [0.0] * len(chain.links)
    for i, deg in enumerate(angles_deg): pose[i + 2] = float(np.radians(deg))
    return chain.forward_kinematics(pose)[:3, 3] * 1000.0


# ── camera + perception ──────────────────────────────────────────────

ROBOT_EXCLUSION_RADIUS_PX = 150   # masks pump tip only — lets box/hand detections survive


def project_base_to_pixel(point_base_mm, mtx, dist, T_cam2base):
    """Project a 3D base-frame point to image pixel coords."""
    T_base2cam = np.linalg.inv(T_cam2base)
    p_base = np.array([*point_base_mm, 1.0])
    p_cam = (T_base2cam @ p_base)[:3]
    if p_cam[2] <= 0:
        return None
    pts = p_cam.reshape(1, 1, 3).astype(np.float32)
    projected, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), mtx, dist)
    return projected[0][0]


def _filter_robot_overlap(candidates, mc, mtx, dist, T_cam2base):
    """Drop candidates whose bbox center is near where the robot's TCP OR
    fixed base structure appears in the image."""
    exclusion = []
    # 1. Current TCP (where the pump/cube is now)
    try:
        tcp = mc.get_coords()
        if isinstance(tcp, (list, tuple)) and len(tcp) >= 3:
            p = project_base_to_pixel(tcp[:3], mtx, dist, T_cam2base)
            if p is not None:
                exclusion.append((float(p[0]), float(p[1]), ROBOT_EXCLUSION_RADIUS_PX))
    except Exception:
        pass
    # 2. Robot base (always at origin; the chunky white base looks hand-like)
    base_p = project_base_to_pixel([0, 0, 60], mtx, dist, T_cam2base)
    if base_p is not None:
        exclusion.append((float(base_p[0]), float(base_p[1]),
                          ROBOT_EXCLUSION_RADIUS_PX))

    if not exclusion:
        return candidates
    out = []
    for bb, sc in candidates:
        cx = (bb[0] + bb[2]) / 2
        cy = (bb[1] + bb[3]) / 2
        blocked = False
        for rx, ry, rad in exclusion:
            if ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 < rad:
                blocked = True
                break
        if not blocked:
            out.append((bb, sc))
    return out


def pixel_to_base_at_z(u, v, mtx, dist, T_cam2base, z_target):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float32)
    norm = cv2.undistortPoints(pts, mtx, dist).reshape(2)
    d = np.array([norm[0], norm[1], 1.0]); d /= np.linalg.norm(d)
    origin = T_cam2base[:3, 3]
    dir_b = T_cam2base[:3, :3] @ d
    if abs(dir_b[2]) < 1e-6: return None
    t = (z_target - origin[2]) / dir_b[2]
    if t < 0: return None
    return origin + t * dir_b


# ── brain ─────────────────────────────────────────────────────────────

ACTION_VOCAB = {"pick", "place", "pick_and_place", "touch", "home", "quit"}

# Settling config for the "touch" action — the script tracks the target,
# only lowers when the target has been stable for STABLE_FRAMES_NEEDED frames.
STABLE_THRESHOLD_MM = 40       # hand tremor / OWL bbox jitter is up to ~30mm
STABLE_THRESHOLD_HAND_MM = 100 # for hands: resist OWL "creeping" up the arm
STABLE_FRAMES_NEEDED = 2       # used for non-hand static targets (cubes/bowls)
STABLE_HAND_SECONDS = 2.0      # for HANDS: release after 2s of stillness; voice "drop" overrides
STABLE_BOX_SECONDS = 3.0       # time to position the container (motion resets timer)
MAX_APPROACH_ATTEMPTS = 200    # raised — was burning out in ~6s when OWL missed frames
MAX_APPROACH_WALL_SECONDS = 60 # also bail if 60s have passed regardless of iteration count
APPROACH_AREA_KEEP = 0.70      # reject partial detections (arm covering target)
APPROACH_AREA_DECAY = 0.995

QWEN_SYSTEM = """You convert the user's robot command into a strict JSON plan.

Output ONLY a single JSON object. Schema depends on action:

For "pick", "place", "touch":
  {"action": "...", "object": "<noun phrase>"}

For "pick_and_place":
  {"action": "pick_and_place", "source": "<source>", "target": "<destination>"}

For "home", "quit":
  {"action": "...", "object": null}

Action meanings:
  "pick"           = grab the object with the suction pump and lift it
  "place"          = put the currently held object onto/above the target;
                     ONLY use when the user said "it" / "the cube" (already held)
                     — e.g. "drop it in the bowl", "put it on my hand"
  "pick_and_place" = pick the source, then place on the target — use whenever
                     BOTH the source and destination are named in one command,
                     e.g. "place the green cube in the box", "hand me the X",
                     "put the pink cube on the yellow one", "give me the blue one".
                     If you see TWO objects mentioned, it's ALWAYS pick_and_place.
  "touch"          = lower pump to just above target without suction (demo)

Examples:
user: "pick the pink cube"               → {"action":"pick","object":"a small pink object"}
user: "pick the green cube"              → {"action":"pick","object":"a green cube"}
user: "pick the blue cube"               → {"action":"pick","object":"a blue cube"}
user: "grab the yellow one"              → {"action":"pick","object":"a small yellow object"}
user: "go to the yellow one"             → {"action":"touch","object":"a small yellow object"}
user: "put it on my hand"                → {"action":"place","object":"a human hand"}
user: "give it to me"                    → {"action":"place","object":"a human hand"}
user: "drop it in the bowl"              → {"action":"place","object":"a bowl"}
user: "place it on the yellow cube"      → {"action":"place","object":"a small yellow object"}
user: "touch the pink cube"              → {"action":"touch","object":"a small pink object"}
user: "back to home" / "reset"           → {"action":"home","object":null}
user: "stop" / "quit"                    → {"action":"quit","object":null}

pick_and_place examples (combine source AND destination in one command):
user: "hand me the green cube"           → {"action":"pick_and_place","source":"a green cube","target":"a human hand"}
user: "give me the pink one"             → {"action":"pick_and_place","source":"a small pink object","target":"a human hand"}
user: "bring me the yellow cube"         → {"action":"pick_and_place","source":"a small yellow object","target":"a human hand"}
user: "put the blue cube in the bowl"    → {"action":"pick_and_place","source":"a blue cube","target":"a bowl"}
user: "drop the green one in the box"    → {"action":"pick_and_place","source":"a green cube","target":"a cardboard box"}
user: "stack the pink on the yellow"     → {"action":"pick_and_place","source":"a small pink object","target":"a small yellow object"}
user: "pick the credit card"             → {"action":"pick","object":"a credit card"}
user: "hand me the credit card"          → {"action":"pick_and_place","source":"a credit card","target":"a human hand"}
user: "give me the bank card"            → {"action":"pick_and_place","source":"a credit card","target":"a human hand"}

Rewrite the object as a noun phrase OWL-ViT can detect. A bare color reference
("the yellow one") means a colored cube.

Color aliasing rule: the user's "purple", "magenta", or "violet" cube is
visually pink under our camera. Always normalize these colors to "pink".
"teal", "cyan", "navy" → "blue".  "lime", "olive" → "green".
  user: "pick the purple cube"   → {"action":"pick","object":"a small pink object"}
  user: "pick the cyan one"      → {"action":"pick","object":"a blue cube"}
  user: "the magenta one"        → {"action":"touch","object":"a small pink object"}

Do not output any other text."""


def qwen_plan(llm, tok, user_text):
    # /no_think suppresses Qwen3's <think> mode → direct JSON output, ~1s
    prompt = (f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{user_text} /no_think<|im_end|>\n"
              f"<|im_start|>assistant\n")
    raw = generate_llm(llm, tok, prompt=prompt, max_tokens=120, verbose=False)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Qwen output had no JSON: {raw!r}")
    plan = json.loads(m.group(0))
    if plan.get("action") not in ACTION_VOCAB:
        raise ValueError(f"unknown action: {plan!r}")
    return plan


# ── execution ────────────────────────────────────────────────────────

def capture(cap):
    for _ in range(5): cap.read()
    ret, frame = cap.read()
    return frame if ret else None


def show_preview(frame, query, box=None, score=None, label=None):
    """Draw OWL bbox on the frame and show it in a window."""
    disp = frame.copy()
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 3)
        text = f"{query} {score:.2f}" if score is not None else query
        cv2.putText(disp, text, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(disp, f"no match for: {query}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if label:
        cv2.putText(disp, label, (20, disp.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    _draw_stats_overlay(disp)
    _update_fps()
    cv2.imshow("robot_view", disp)
    # Pump multiple waitKey ticks so macOS marks the window as actively
    # updating — otherwise it freezes during 500ms+ OWL inferences.
    for _ in range(5):
        cv2.waitKey(1)


def _filter_oversized(boxes, scores, frame_area):
    """Keep only detections whose bbox is small enough to plausibly be the
    target (not the entire desk). Returns lists of (box, score) tuples."""
    out = []
    for b, s in zip(boxes, scores):
        bb = b.tolist()
        area = max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])
        if area / frame_area <= OWL_MAX_BBOX_AREA_FRAC:
            out.append((bb, float(s)))
    return out


CUBE_COLORS_KNOWN = ["green", "blue", "pink", "yellow", "red"]
CARD_QUERIES = ["a credit card", "an ID card", "a driver's license"]


def _disambiguation_queries(query):
    """If the target falls into a known confusable category (colored cube, or
    a specific card type), return (all_alternative_queries, exact_target_query)
    so OWL scores each alternative separately and find_object keeps only the
    detections OWL labeled as the requested one.
    Otherwise returns ([query], None) — no disambiguation."""
    q = query.lower()
    # Cube colors — keep existing behavior
    for c in CUBE_COLORS_KNOWN:
        if c in q and "cube" in q:
            return [f"a {col} cube" for col in CUBE_COLORS_KNOWN], f"a {c} cube"
    # Card types — credit/ID/license all look like rectangular plastic to OWL
    if "credit" in q and "card" in q:
        return CARD_QUERIES, "a credit card"
    if "id card" in q or "identification" in q:
        return CARD_QUERIES, "an ID card"
    if ("driver" in q or "driving" in q) and ("license" in q or "licence" in q):
        return CARD_QUERIES, "a driver's license"
    return [query], None


def find_object(frame, query, processor, model, device, mtx, dist, T_cam2base, z_plane):
    queries, target_query = _disambiguation_queries(query)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=[queries], images=pil, return_tensors="pt").to(device)
    _t0 = time.time()
    with torch.no_grad(): outputs = model(**inputs)
    _stats["owl_ms"] = (time.time() - _t0) * 1000
    _log_owl("find_object", _stats["owl_ms"], query)
    res = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=torch.Tensor([pil.size[::-1]]),
        threshold=_owl_threshold_for(query))[0]
    frame_area = frame.shape[0] * frame.shape[1]

    # If we disambiguated, prefer detections OWL labeled with the target
    # query. If none, fall back to other-labeled alternatives so we don't
    # miss a detection just because OWL confused similar-looking objects.
    if target_query is not None:
        target_idx = queries.index(target_query)
        target_b, target_s = [], []
        other_b, other_s = [], []
        for b, s, lab in zip(res["boxes"], res["scores"], res["labels"]):
            if int(lab) == target_idx:
                target_b.append(b); target_s.append(s)
            else:
                other_b.append(b); other_s.append(s)
                dlog(f"find_object alt-label: queried '{target_query}' got '{queries[int(lab)]}' conf={float(s):.2f}")
        if target_b:
            filtered_boxes, filtered_scores = target_b, target_s
        else:
            dlog(f"find_object FALLBACK: no '{target_query}' labels, using {len(other_b)} alt-labeled detections")
            filtered_boxes, filtered_scores = other_b, other_s
        candidates = _filter_oversized(filtered_boxes, filtered_scores, frame_area)
    else:
        candidates = _filter_oversized(res["boxes"], res["scores"], frame_area)

    if not candidates:
        show_preview(frame, query)
        return None, None
    box, score = max(candidates, key=lambda c: c[1])
    cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
    show_preview(frame, query, box, score)
    base = pixel_to_base_at_z(cx, cy, mtx, dist, T_cam2base, z_plane)
    if base is None: return None, None
    return base[:2], score


def move_to_tcp(mc, target_xyz_mm, label, require_down=False):
    current = mc.get_angles()
    if not isinstance(current, (list, tuple)) or len(current) != 6:
        print(f"  [{label}] bad get_angles read: {current!r}"); return False
    # Try pump-down first; fall back to free orientation if unreachable
    # UNLESS the caller requires straight-down (e.g., for suction pickup).
    mode = "down"
    try:
        angles = solve_ik(target_xyz_mm, current, pointing_down=True)
    except Exception as e:
        print(f"  [{label}] IK exc (down): {e}"); return False
    err = float(np.linalg.norm(fk_pos_mm(angles) - target_xyz_mm))
    if err > IK_ERR_LIMIT_MM:
        if require_down:
            print(f"  [{label}] unreachable with pump straight down (err={err:.0f}mm) — refusing tilt fallback")
            say("out_of_reach"); return False
        try:
            angles = solve_ik(target_xyz_mm, current, pointing_down=False)
        except Exception as e:
            print(f"  [{label}] IK exc (free): {e}"); return False
        err = float(np.linalg.norm(fk_pos_mm(angles) - target_xyz_mm))
        mode = "tilted"
        if err > IK_ERR_LIMIT_MM:
            print(f"  [{label}] unreachable (err={err:.0f}mm)"); say("out_of_reach"); return False
    swing = max(abs(a - b) for a, b in zip(angles, current))
    if swing > MAX_JOINT_STEP_DEG:
        print(f"  [{label}] joint swing too large ({swing:.0f}°)"); return False
    mc.send_angles(angles, SPEED)
    if interruptible_sleep(2.5, mc):
        print(f"  [{label}] interrupted mid-move")
        return False
    print(f"  [{label}] TCP={target_xyz_mm.round(1)}  err={err:.0f}mm  swing={swing:.0f}°  {mode}")
    return True


def smart_touch(mc, cap, processor, model, device, mtx, dist, T_cam2base,
                obj, z_plane_mm):
    """Track target until it stops moving, then lower onto it.

    Approach phase: each frame, detect object → if XY drifted > STABLE_THRESHOLD_MM,
    move arm above the new XY and reset stability counter. When the target
    holds still for STABLE_FRAMES_NEEDED frames, commit and lower.
    """
    print(f"  ▶ approaching '{obj}' — waiting for it to hold still")

    locked_xy = None
    stable_count = 0
    recent_max_area = 0.0
    attempts = 0

    try:
        while attempts < MAX_APPROACH_ATTEMPTS:
            if check_interrupt():
                print("  ▶ stopped by user"); say("stopping")
                try: mc.stop()
                except Exception: pass
                return
            attempts += 1
            frame = capture(cap)
            if frame is None: continue

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(text=[[obj]], images=pil, return_tensors="pt").to(device)
            _t0 = time.time()
            with torch.no_grad(): outputs = model(**inputs)
            _stats["owl_ms"] = (time.time() - _t0) * 1000
            _log_owl("smart_touch", _stats["owl_ms"], obj)
            res = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=torch.Tensor([pil.size[::-1]]),
                threshold=_owl_threshold_for(obj))[0]
            frame_area = frame.shape[0] * frame.shape[1]
            candidates = _filter_oversized(res["boxes"], res["scores"], frame_area)
            candidates = _filter_robot_overlap(candidates, mc, mtx, dist, T_cam2base)
            if not candidates:
                show_preview(frame, obj, None, None, "searching…")
                if attempts % 8 == 0:
                    print(f"    (searching — no '{obj}' detected, {attempts}/{MAX_APPROACH_ATTEMPTS}) — say 'drop' to release now")
                continue
            box, score = max(candidates, key=lambda c: c[1])
            x1, y1, x2, y2 = box
            area = max(0, x2 - x1) * max(0, y2 - y1)
            recent_max_area = max(area, recent_max_area * APPROACH_AREA_DECAY)
            if area < recent_max_area * APPROACH_AREA_KEEP:
                show_preview(frame, obj, box, score, label=f"partial — ignored")
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            base = pixel_to_base_at_z(cx, cy, mtx, dist, T_cam2base, z_plane_mm)
            if base is None: continue
            xy = np.array(base[:2])

            # Compare to locked position
            if locked_xy is None:
                locked_xy = xy
                stable_count = 1
                hover_tcp = np.array([xy[0], xy[1],
                                      z_plane_mm + HOVER_ABOVE + PUMP_LENGTH])
                show_preview(frame, obj, box, score, "approaching")
                move_to_tcp(mc, hover_tcp, "approach")
                continue

            drift = float(np.linalg.norm(xy - locked_xy))
            if drift > STABLE_THRESHOLD_MM:
                print(f"    target moved ({drift:.0f}mm) — re-approaching")
                locked_xy = xy
                stable_count = 1
                hover_tcp = np.array([xy[0], xy[1],
                                      z_plane_mm + HOVER_ABOVE + PUMP_LENGTH])
                show_preview(frame, obj, box, score, "re-approach (moved)")
                move_to_tcp(mc, hover_tcp, "approach")
                continue

            stable_count += 1
            show_preview(frame, obj, box, score, f"stable {stable_count}/{STABLE_FRAMES_NEEDED}")
            print(f"    stable {stable_count}/{STABLE_FRAMES_NEEDED}  drift={drift:.0f}mm")
            if stable_count >= STABLE_FRAMES_NEEDED:
                # commit and lower
                place_tcp = np.array([xy[0], xy[1],
                                      z_plane_mm + TOUCH_ABOVE + PUMP_LENGTH])
                hover_tcp = np.array([xy[0], xy[1],
                                      z_plane_mm + HOVER_ABOVE + PUMP_LENGTH])
                print(f"  ▶ target stable — placing")
                move_to_tcp(mc, place_tcp, "place")
                time.sleep(0.6)
                move_to_tcp(mc, hover_tcp, "lift")
                return

        print(f"  ⚠ target never settled (gave up after {MAX_APPROACH_ATTEMPTS} frames)")
    except KeyboardInterrupt:
        print("\n  ▶ interrupted — turning pump OFF")
        try: pump_off(mc)
        except Exception: pass
        try: mc.stop()
        except Exception: pass


PICK_SEARCH_MAX_ATTEMPTS = 60   # ~30s of searching at ~2 fps before giving up


def do_pick(mc, cap, processor, model, device, mtx, dist, T_cam2base, obj):
    """Static-target pick: detect (retrying until found, interrupted, or
    timeout), hover, lower, pump on, lift.
    Returns True on success, False on any failure."""
    # Cards / paper / coins lie flat on the desk; need to descend almost to the
    # table surface. Cubes need to stop at the cube top.
    is_flat = any(w in obj.lower() for w in ("card", "coin", "paper", "sticker"))
    z_plane = TABLE_Z_BASE_MM if is_flat else TABLE_Z_BASE_MM + CUBE_HEIGHT_MM

    # ── search loop: keep looking until OWL finds it ──
    xy, score = None, None
    print(f"  ▶ looking for '{obj}' — say 'stop' to give up")
    for attempt in range(PICK_SEARCH_MAX_ATTEMPTS):
        if check_interrupt():
            print("  ▶ search cancelled by user"); say("stopping"); return False
        frame = capture(cap)
        if frame is None: continue
        xy, score = find_object(frame, obj, processor, model, device, mtx, dist,
                                T_cam2base, z_plane)
        if xy is not None:
            break
        if attempt > 0 and attempt % 8 == 0:
            print(f"    still looking... ({attempt}/{PICK_SEARCH_MAX_ATTEMPTS})")
    if xy is None:
        print(f"  '{obj}' not found after {PICK_SEARCH_MAX_ATTEMPTS} attempts")
        say("not_found")
        return False
    print(f"  picking '{obj}' at ({xy[0]:.0f}, {xy[1]:.0f}) conf={score:.2f}")
    say("picking")

    # For flat targets (card/coin/paper), descend below the assumed z=0 plane
    # since the real desk is a few mm below where calibration thinks it is.
    # For cubes, TOUCH_ABOVE=-5 presses into the rubber for a firm suction seal.
    touch = -10.0 if is_flat else TOUCH_ABOVE
    pick_tip   = np.array([xy[0], xy[1], z_plane + touch])
    hover_tip  = np.array([xy[0], xy[1], z_plane + HOVER_ABOVE])
    pick_tcp  = pick_tip + np.array([0, 0, PUMP_LENGTH])
    hover_tcp = hover_tip + np.array([0, 0, PUMP_LENGTH])

    if not move_to_tcp(mc, hover_tcp, "approach"): return False
    # Flat targets (cards) need a perfect straight-down seal; cubes tolerate
    # a slight tilt because the rubber cup deforms around the cube top.
    if not move_to_tcp(mc, pick_tcp, "down", require_down=True):
        # Failed mid-pick: arm is partly extended without a cube. Return to
        # a safer pose (the approach hover) instead of leaving it stranded.
        move_to_tcp(mc, hover_tcp, "retreat")
        return False
    pump_on(mc)
    print("  pump ON")
    say("got_it")
    if interruptible_sleep(PUMP_DWELL_S, mc):
        print("  interrupted — pump OFF")
        pump_off(mc); return False
    if not move_to_tcp(mc, hover_tcp, "lift"):
        # interrupted/rejected during lift — keep pump engaged so the cube
        # isn't dropped on the desk
        return False
    print("  picked")
    return True


def do_place(mc, cap, processor, model, device, mtx, dist, T_cam2base, obj):
    """Dynamic-target place: track until target settles, then release.

    If the target is a hand (held up by the user, not on the table), we
    release at hover height — no descent — and let the cube drop into the hand.
    For static targets (cubes), we lower to just above the target before release.
    """
    from collections import deque
    MEDIAN_WINDOW = 5
    DETECTION_GAP_RESET_S = 1.5
    PARTIAL_AREA_THRESHOLD = 0.55  # if a post-lock detection is < this × locked area,
                                    # treat as occlusion (arm covering box), don't retarget
    recent_xys = deque(maxlen=MEDIAN_WINDOW)
    # Hand is typically held above the desk; using the table z plane for
    # back-projection makes the XY estimate skewed when the hand is at the
    # edge of the camera frame. Use a higher z plane for hand targets.
    is_hand_check = any(w in obj.lower() for w in ("hand", "palm", "finger"))
    z_plane = HAND_Z_BASE_MM if is_hand_check else TABLE_Z_BASE_MM + CUBE_HEIGHT_MM
    # Hand uses a much smaller hover so the TCP stays within the 280mm reach.
    hover_above = HAND_HOVER_ABOVE if is_hand_check else HOVER_ABOVE
    locked_xy_time = 0.0
    last_in_cluster_time = 0.0
    locked_area = 0.0          # area of the bbox when we first acquired the lock
    # Targets to release AT HOVER (no descent): hands and containers.
    # For a static cube target we still descend to TOUCH_ABOVE.
    is_hand = any(w in obj.lower() for w in ("hand", "palm", "finger"))
    drop_in = is_hand or any(w in obj.lower()
                             for w in ("bowl", "cup", "container", "tray", "box"))
    if is_hand:
        print(f"  ▶ chasing '{obj}'")
        print(f"     say 'DROP' anytime to release the cube now")
        print(f"     OR hold hand still for {STABLE_HAND_SECONDS:.0f}s and it'll auto-release")
    else:
        print(f"  ▶ placing on '{obj}'{' (release at hover, drop in)' if drop_in else ''}"
              f" — waiting for it to hold still")
    say("placing")

    locked_xy = None
    stable_count = 0
    recent_max_area = 0.0
    attempts = 0

    try:
        wall_start = time.time()
        while attempts < MAX_APPROACH_ATTEMPTS and (time.time() - wall_start) < MAX_APPROACH_WALL_SECONDS:
            if check_interrupt():
                print("  ▶ stopped by user (cube NOT released)")
                try: mc.stop()
                except Exception: pass
                return

            # Voice "drop" works even without a fresh OWL detection
            if locked_xy is not None and check_release_request():
                print("  ▶ user said release — dropping at locked position")
                hover_tcp = np.array([locked_xy[0], locked_xy[1],
                                      z_plane + hover_above + PUMP_LENGTH])
                pump_off(mc); print("  pump OFF — released"); say("released")
                time.sleep(0.4)
                move_to_tcp(mc, hover_tcp, "lift")
                return

            # Wall-time auto-release for hands and containers — works under occlusion.
            # Trust the approach: cube goes where we last detected the target.
            if drop_in and locked_xy is not None and locked_xy_time > 0:
                threshold_s = STABLE_HAND_SECONDS if is_hand else STABLE_BOX_SECONDS
                stillness_s = time.time() - locked_xy_time
                if stillness_s >= threshold_s:
                    print(f"  ▶ auto-release after {stillness_s:.1f}s (target: {obj})")
                    hover_tcp = np.array([locked_xy[0], locked_xy[1],
                                          z_plane + hover_above + PUMP_LENGTH])
                    pump_off(mc); print("  pump OFF — released"); say("released")
                    time.sleep(0.6)
                    move_to_tcp(mc, hover_tcp, "lift")
                    return

            attempts += 1
            frame = capture(cap)
            if frame is None: continue

            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Color-disambiguate: when the target is a colored cube, send all
            # known color queries so OWL labels each detection by color, then
            # we keep only the target color.
            owl_queries, target_query = _disambiguation_queries(obj)
            inputs = processor(text=[owl_queries], images=pil, return_tensors="pt").to(device)
            _t0 = time.time()
            with torch.no_grad(): outputs = model(**inputs)
            _stats["owl_ms"] = (time.time() - _t0) * 1000
            _log_owl("do_place", _stats["owl_ms"], obj)
            res = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=torch.Tensor([pil.size[::-1]]),
                threshold=_owl_threshold_for(obj))[0]
            # If we disambiguated, keep only detections labeled with target query
            if target_query is not None:
                target_idx = owl_queries.index(target_query)
                kept_b = []; kept_s = []
                for b, s, lab in zip(res["boxes"], res["scores"], res["labels"]):
                    if int(lab) == target_idx:
                        kept_b.append(b); kept_s.append(s)
                res = {"boxes": kept_b, "scores": kept_s}
            raw_n = len(res["boxes"])
            frame_area = frame.shape[0] * frame.shape[1]
            try:
                tcp_now = mc.get_coords()
                if not isinstance(tcp_now, (list, tuple)) or len(tcp_now) < 3:
                    tcp_now = [0, 0, 0]
            except Exception:
                tcp_now = [0, 0, 0]
            tcp_proj = project_base_to_pixel(tcp_now[:3], mtx, dist, T_cam2base)
            tcp_proj_str = f"({int(tcp_proj[0])},{int(tcp_proj[1])})" if tcp_proj is not None else "None"
            for b, s in zip(res["boxes"], res["scores"]):
                bb = b.tolist() if hasattr(b, "tolist") else b; sc = float(s)
                cx_d = int((bb[0]+bb[2])/2); cy_d = int((bb[1]+bb[3])/2)
                area_d = int(max(0, bb[2]-bb[0]) * max(0, bb[3]-bb[1]))
                dlog(f"do_place RAW: q={obj!r} conf={sc:.3f} center=({cx_d},{cy_d}) area={area_d} frac={area_d/frame_area:.3f}")
            candidates = _filter_oversized(res["boxes"], res["scores"], frame_area)
            after_size = len(candidates)
            candidates = _filter_robot_overlap(candidates, mc, mtx, dist, T_cam2base)
            after_robot = len(candidates)
            dlog(f"do_place FILTER: raw={raw_n} after_oversize={after_size} after_robot={after_robot} tcp={tcp_now[:3]} tcp_proj_px={tcp_proj_str}")
            if not candidates:
                show_preview(frame, obj, None, None, "searching…")
                if attempts % 8 == 0:
                    print(f"    (searching — no '{obj}' detected, {attempts}/{MAX_APPROACH_ATTEMPTS}) — say 'drop' to release now")
                continue
            box, score = max(candidates, key=lambda c: c[1])
            x1, y1, x2, y2 = box
            area = max(0, x2 - x1) * max(0, y2 - y1)

            # AFTER lock: if the current detection's area is much smaller than
            # what we locked onto, it's likely the arm partially occluding the
            # target. Ignore it as a "position update" but treat as a confirmation
            # that the target is still where we last saw it (so the stillness
            # timer keeps ticking and we don't get stuck).
            if locked_xy is not None and locked_area > 0 and area < locked_area * PARTIAL_AREA_THRESHOLD:
                dlog(f"do_place PARTIAL (occluded): area={area:.0f}<{locked_area*PARTIAL_AREA_THRESHOLD:.0f} — keeping lock")
                show_preview(frame, obj, box, score,
                             f"box partly hidden — holding position")
                # Treat as in-cluster: timer keeps counting toward auto-release
                last_in_cluster_time = time.time()
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            base = pixel_to_base_at_z(cx, cy, mtx, dist, T_cam2base, z_plane)
            if base is None:
                dlog(f"do_place REJECTED: back-projection failed for ({int(cx)},{int(cy)})")
                continue
            xy_raw = np.array(base[:2])
            dlog(f"do_place ACCEPT: pixel=({int(cx)},{int(cy)}) base_xy=({xy_raw[0]:.0f},{xy_raw[1]:.0f}) conf={score:.3f}")

            recent_xys.append(xy_raw)
            # Use raw position on the first detection so we move immediately.
            # Once we have 2+ samples the median smooths jitter on subsequent frames.
            if len(recent_xys) >= 2:
                xy = np.median(np.array(recent_xys), axis=0)
            else:
                xy = xy_raw
            dlog(f"do_place POS: xy=({xy[0]:.0f},{xy[1]:.0f}) raw=({xy_raw[0]:.0f},{xy_raw[1]:.0f}) samples={len(recent_xys)}")

            # Use larger threshold for hands to resist OWL's "creep" (detection
            # drifting from palm to wrist to forearm as arm covers the palm)
            threshold_mm = STABLE_THRESHOLD_HAND_MM if is_hand else STABLE_THRESHOLD_MM
            # Detect move on EITHER raw or median position — raw catches fast
            # moves before the median catches up.
            moved_raw = locked_xy is not None and float(np.linalg.norm(xy_raw - locked_xy)) > threshold_mm
            moved_med = locked_xy is not None and float(np.linalg.norm(xy - locked_xy)) > threshold_mm
            if locked_xy is None or moved_raw or moved_med:
                if locked_xy is not None:
                    print(f"    target moved — re-approaching (raw={'yes' if moved_raw else 'no'} med={'yes' if moved_med else 'no'})")
                # On a new lock, use raw position (skips median lag) and clear
                # the buffer so old positions don't drag the next median.
                new_lock_xy = xy_raw
                recent_xys.clear()
                recent_xys.append(xy_raw)
                locked_xy = new_lock_xy
                locked_area = area
                stable_count = 1
                hover_tcp = np.array([new_lock_xy[0], new_lock_xy[1],
                                      z_plane + hover_above + PUMP_LENGTH])
                show_preview(frame, obj, box, score, "approaching")
                if not move_to_tcp(mc, hover_tcp, "approach"):
                    print("  ▶ approach interrupted — aborting place action")
                    return
                locked_xy_time = time.time()
                last_in_cluster_time = time.time()
                dlog(f"do_place NEW LOCK: xy={[int(v) for v in new_lock_xy]} area={int(area)}")
                continue

            stable_count += 1
            now = time.time()
            last_in_cluster_time = now
            stillness_s = now - locked_xy_time
            if is_hand:
                show_preview(frame, obj, box, score,
                             f"in position — releasing in {STABLE_HAND_SECONDS - stillness_s:.1f}s")
                print(f"    in position — releasing in {STABLE_HAND_SECONDS - stillness_s:.1f}s")
                pump_preview(0.4, label=f"releasing in {STABLE_HAND_SECONDS - stillness_s:.1f}s")
            else:
                show_preview(frame, obj, box, score,
                             f"stable {stable_count}/{STABLE_FRAMES_NEEDED}")
                print(f"    stable {stable_count}/{STABLE_FRAMES_NEEDED}  median@{xy.round(0)}")
            # Voice override always wins
            forced = check_release_request()
            if forced:
                print("  ▶ user said release — forcing drop now")
            # Hand → release after STABLE_HAND_SECONDS of stillness
            # Container (box/bowl) → release after STABLE_BOX_SECONDS of stillness
            # Static cube target → release after STABLE_FRAMES_NEEDED stable frames
            ready_to_release = forced or (
                (is_hand and stillness_s >= STABLE_HAND_SECONDS)
                or (drop_in and not is_hand and stillness_s >= STABLE_BOX_SECONDS)
                or (not drop_in and stable_count >= STABLE_FRAMES_NEEDED)
            )
            if ready_to_release:
                hover_tcp = np.array([xy[0], xy[1],
                                      z_plane + hover_above + PUMP_LENGTH])
                if drop_in:
                    # Hand catches the cube at hover height — no descent.
                    print("  ▶ releasing at hover height (no descent)")
                    pump_off(mc)
                    print("  pump OFF — released into hand")
                    time.sleep(0.6)
                    move_to_tcp(mc, hover_tcp, "lift")  # small re-affirm move
                else:
                    place_tcp = np.array([xy[0], xy[1],
                                          z_plane + TOUCH_ABOVE + PUMP_LENGTH])
                    print("  ▶ lowering to place")
                    if not move_to_tcp(mc, place_tcp, "down"):
                        pump_off(mc); return
                    pump_off(mc)
                    print("  pump OFF — released"); say("released")
                    time.sleep(0.4)
                    move_to_tcp(mc, hover_tcp, "lift")
                return

        print(f"  ⚠ target never settled — keeping object")
    except KeyboardInterrupt:
        print("\n  ▶ interrupted — turning pump OFF")
        try: pump_off(mc)
        except Exception: pass
        try: mc.stop()
        except Exception: pass


def live_preview(cap, processor, model, device, query=None, show_all=False):
    """Live cam preview with optional OWL overlay.
    If show_all=True, draws every detection (rejected ones in red), not just
    the chosen one. Press 'q' to close."""
    print(f"  live preview — query={query!r}  show_all={show_all}  ('q' to close)")
    last_owl = 0.0
    last_dets = []   # list of (box, score, rejected_reason or None)
    try:
        while True:
            ret, frame = cap.read()
            if not ret: time.sleep(0.05); continue
            disp = frame.copy()
            frame_area = frame.shape[0] * frame.shape[1]
            now = time.time()
            if query and (now - last_owl) > 0.5:
                last_owl = now
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = processor(text=[[query]], images=pil, return_tensors="pt").to(device)
                _t0 = time.time()
                with torch.no_grad(): outputs = model(**inputs)
                _stats["owl_ms"] = (time.time() - _t0) * 1000
                _log_owl("live_preview", _stats["owl_ms"], query)
                res = processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=torch.Tensor([pil.size[::-1]]),
                    threshold=_owl_threshold_for(query))[0]
                dets = []
                for b, s in zip(res["boxes"], res["scores"]):
                    bb = b.tolist(); sc = float(s)
                    area = max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])
                    frac = area / frame_area
                    rej = "oversized" if frac > OWL_MAX_BBOX_AREA_FRAC else None
                    dets.append((bb, sc, frac, rej))
                last_dets = dets

            kept = [d for d in last_dets if d[3] is None]
            chosen = max(kept, key=lambda d: d[1]) if kept else None

            for bb, sc, frac, rej in last_dets:
                x1, y1, x2, y2 = [int(v) for v in bb]
                if rej == "oversized":
                    if show_all:
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(disp, f"rej {sc:.2f} ({frac*100:.0f}%)",
                                    (x1, max(y1 - 8, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                elif (bb, sc, frac, rej) == chosen:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(disp, f"{query} {sc:.2f} ({frac*100:.0f}%)",
                                (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif show_all:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 200, 0), 1)
                    cv2.putText(disp, f"{sc:.2f} ({frac*100:.0f}%)",
                                (x1, max(y1 - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            if chosen is None and query:
                cv2.putText(disp, f"no match: {query}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            help_txt = "q=close" + ("  (showing all detections)" if show_all else "")
            cv2.putText(disp, help_txt, (20, disp.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            _draw_stats_overlay(disp)
            _update_fps()
            cv2.imshow("robot_view", disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cv2.destroyWindow("robot_view")
        for _ in range(5): cv2.waitKey(1)


def execute(plan, mc, cap, processor, model, device, mtx, dist, T_cam2base):
    action = plan["action"]
    if action == "home":
        print("  → homing"); say("going_home")
        mc.send_angles(HOME_ANGLES, SPEED); time.sleep(4)
        return
    if action == "quit":
        return
    if action == "pick_and_place":
        source = plan.get("source")
        target = plan.get("target")
        if not source or not target:
            print("  pick_and_place needs both 'source' and 'target'"); return
        print(f"  ▶ chained: pick '{source}' then place on '{target}'")
        ok = do_pick(mc, cap, processor, model, device, mtx, dist, T_cam2base, source)
        if not ok:
            print(f"  ▶ pick failed — aborting chain (place phase skipped)")
            return
        if check_interrupt():
            print("  ▶ interrupted between pick and place"); return
        do_place(mc, cap, processor, model, device, mtx, dist, T_cam2base, target)
        return
    obj = plan.get("object")
    if not obj:
        print("  no object given"); return

    if action == "pick":
        do_pick(mc, cap, processor, model, device, mtx, dist, T_cam2base, obj)
    elif action == "place":
        if not _pump_engaged:
            print(f"  ⚠ Pump isn't holding anything — 'place' requires a held cube.")
            print(f"  ⚠ If you wanted to deliver a cube to {obj!r}, say:")
            print(f"  ⚠   'Drop the [color] cube in the {obj.split(chr(32),1)[-1] if ' ' in obj else obj}'")
            print(f"  ⚠   'Hand me the [color] cube' (for hand delivery)")
            return
        do_place(mc, cap, processor, model, device, mtx, dist, T_cam2base, obj)
    elif action == "touch":
        z_plane = TABLE_Z_BASE_MM + CUBE_HEIGHT_MM
        smart_touch(mc, cap, processor, model, device, mtx, dist, T_cam2base,
                    obj, z_plane_mm=z_plane)


# ── main ──────────────────────────────────────────────────────────────

def main():
    with open(CALIB_PATH) as f: cal = json.load(f)
    mtx = np.array(cal["camera_matrix"])
    dist = np.array(cal["dist_coeffs"])
    T_cam2base = np.array(cal["T_cam2base"])

    print("Loading Qwen3-1.7B...")
    llm, tok = load_llm("Qwen/Qwen3-1.7B")
    print("Loading OWL-ViT...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

    _open_metrics_log()
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    global _live_cap
    _live_cap = cap   # let interruptible_sleep refresh the preview while we wait

    print("Connecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2); mc.power_on(); time.sleep(0.5)
    print("Homing arm to [0,0,0,0,0,0]...")
    mc.send_angles(HOME_ANGLES, SPEED)
    time.sleep(4)

    print("Loading STT (Nemotron)...")
    # STT model toggle. Set STT_MODEL above the loader to switch.
    #   "nemotron" — original tested model
    #   "parakeet" — English-only, lower hallucination, faster
    STT_MODEL = "nemotron"   # ← change to "parakeet" to try the newer model
    STT_OPTIONS = {
        "nemotron": "mlx-community/nemotron-3.5-asr-streaming-0.6b",
        "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
    }
    print(f"Loading STT ({STT_MODEL}): {STT_OPTIONS[STT_MODEL]}")
    stt = load_stt(STT_OPTIONS[STT_MODEL])
    start_voice(stt)
    print("  ▶ mic is live — speak or type at the prompt\n")

    print("Ready. Voice-only mode. Speak a command. Ctrl+C to exit.\n")
    try:
        while True:
            # Voice-only: poll the queue and refresh the camera preview while idle
            # so the window doesn't freeze between actions.
            try:
                text = voice_queue.get(timeout=0.05)
            except queue.Empty:
                pump_preview(0.15, label="idle — speak a command")
                continue
            text = text.strip()
            if not text:
                continue
            # Idle "stop" → no-op (nothing to interrupt at the prompt)
            if _is_interrupt_text(text):
                continue
            if text.lower() in ("quit", "exit"):
                break

            try:
                plan = qwen_plan(llm, tok, text)
            except Exception as e:
                print(f"  ⚠ plan parse failed: {e}"); say("didnt_understand")
                continue
            print(f"  plan: {plan}"); say("ok")
            if plan["action"] == "quit":
                break
            execute(plan, mc, cap, processor, model, device, mtx, dist, T_cam2base)
    except KeyboardInterrupt:
        pass
    finally:
        _voice_stop.set()
        try: pump_off(mc)
        except Exception: pass
        cap.release()
        print("Done.")


if __name__ == "__main__":
    main()
