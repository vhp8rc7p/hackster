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
SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
CALIB_PATH = "calibration_result.json"
URDF_PATH = "mycobot_280_m5.urdf"

# Input mode: "voice" (mic + STT) or "text" (type commands at the prompt).
INPUT_MODE = "voice"

# IK backend — the toggle for how the arm moves:
#   "native_tool" — firmware IK using the mycobot TOOL FRAME (set_tool_reference
#                   + set_end_type). send_coords drives the PUMP TIP directly and
#                   keeps it vertical. This is the pump-aware method that works in
#                   test_cube_hover.py. RECOMMENDED / default.
#   "native"      — firmware IK targeting the bare FLANGE with a manual pump
#                   offset in Python (older; the arm doesn't "know" the pump).
#   "ikpy"        — pure-Python iterative custom IK (always runs, slow/imperfect)
#   "ikfast"      — analytical from OpenRAVE (needs compiled ikfast_mycobot280)
IK_BACKEND = "native_tool"

# Pixel-to-base mapping mode.
#   "handeye" — 3D hand-eye T_cam2base (works for any Z; supports back-projection
#               to cube tops, hand height, etc.)
#   "affine"  — 2D pixel→base-XY affine (table-plane only; sub-mm fit for flat
#               objects like cards, but Z-mismatch error for cubes/hands)
CALIB_MODE = "affine"             # ← change to "handeye" to swap back
AFFINE_CALIB_PATH = "calibration_affine2d.json"

PUMP_LENGTH = 70.0   # was 50 — your real pump is ~7 cm
# Measured on the arm: the pump seals on the cube TOP at tip Z ≈ 60mm.
# Cube is 25mm tall → table surface ≈ 35mm in the robot base Z frame.
TABLE_Z_BASE_MM = 35.0    # real table height in the robot base Z frame
CUBE_HEIGHT_MM = 25.0     # → cube top (z_plane) = 35 + 25 = 60mm (the seal Z)
HAND_Z_BASE_MM = 135.0   # palm held ~10cm above the desk (table 35 + 100)
HOVER_ABOVE = 55.0        # place/touch/tracking clearance (keep hover reachable
                          # at workspace edge: high Z + far XY exceeds 280mm reach)
PICK_HOVER_ABOVE = 50.0   # pick approach clearance → hover Z ≈ 110mm (reachable);
                          # 100 gave Z=160 which was UNREACHABLE at far cubes
HAND_HOVER_ABOVE = 50.0   # smaller because the user catches the cube; keeps TCP in reach
TOUCH_ABOVE = -5.0   # press 5mm into the cube top for a firm suction seal
                     # → pick tip Z = 60 - 5 = 55mm (measured seal height)
SPEED = 55   # was 30 — motion is ~50% of a pick's wall time
IK_ERR_LIMIT_MM = 30  # accept straight-down solutions up to 30mm off — cup deforms to absorb it. Prevents unnecessary tilt fallback at edge of reach.
MAX_JOINT_STEP_DEG = 180   # allow IK wrist-flips for picks at edge of reach
HOME_ANGLES = [0, 0, 0, 0, 0, 0]

# Suction pump (from pick_aruco.py / ai_pick.py)
PUMP_PIN = 2
VALVE_PIN = 5
PUMP_DWELL_S = 0.8     # time at touch height with pump on before lifting
                       # (was 1.2 — the cup seals well before then)


_pump_engaged = False  # tracks whether the pump is currently holding something


def _wait_for_pending_speech(timeout=2.5):
    """If the user is mid-sentence or we're transcribing, WAIT before doing
    something irreversible (like releasing the cube).

    check_interrupt() only sees text already in the queue. A 'wait' spoken a
    moment earlier is still inside the VAD/STT pipeline, so the release fires
    first and the cube drops anyway. Pausing here lets that text land.
    """
    t0 = time.time()
    while (globals().get("_voice_state", "idle") != "idle"
           and time.time() - t0 < timeout):
        pump_preview(0.1, label="waiting for speech…")
    return check_interrupt()


def _abort_and_raise(mc):
    """Interrupt handler for a place: stop, raise straight up KEEPING the cube,
    and hold for the next command. Used both at the top of the loop and as a
    last-moment guard right before any release, so a late 'wait' still wins."""
    print(t("stopped_not_released"))
    try:
        mc.stop()
        cur = mc.get_coords()
        if isinstance(cur, (list, tuple)) and len(cur) == 6:
            safe_z = min(cur[2] + 80, 125)   # lift ~8cm, capped to stay in reach
            if safe_z > cur[2] + 5:
                _send_and_wait(mc, [cur[0], cur[1], safe_z,
                                    cur[3], cur[4], cur[5]], "wait-raise")
        print("  ▶ raised and holding — say your next command")
    except Exception:
        pass


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
        # Append, not truncate — otherwise every run wipes the history and we
        # can never see steady-state timings across runs.
        new = not os.path.exists(METRICS_LOG)
        _metrics_file = open(METRICS_LOG, "a", buffering=1)  # line-buffered
        if new:
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
            _debug_file = open(DEBUG_LOG, "a", buffering=1)  # append, keep history
            _debug_file.write(f"=== session started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except Exception:
            return
    try:
        _debug_file.write(f"{time.time():.3f}  {msg}\n")
    except Exception:
        pass
_fps_t0 = time.time()
_fps_n = 0


# ── phase timing (for speed optimization) ──────────────────────────────
import contextlib
TIMING_LOG = "timing.csv"
_timing_file = None
_phase_times = []   # (name, seconds) accumulated for the current action


@contextlib.contextmanager
def time_phase(name):
    """Time a block: prints '⏱ name: X.XXs', logs to timing.csv, and records it
    for the per-action summary. Wraps returns/exceptions safely."""
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        _phase_times.append((name, dt))
        print(f"  ⏱ {name}: {dt:.2f}s")
        global _timing_file
        try:
            if _timing_file is None:
                new = not os.path.exists(TIMING_LOG)
                _timing_file = open(TIMING_LOG, "a", buffering=1)
                if new:
                    _timing_file.write("timestamp,phase,seconds\n")
            _timing_file.write(f"{time.time():.3f},{name},{dt:.3f}\n")
        except Exception:
            pass


def timing_summary(label="action"):
    """Print the phases since the last reset and the total, then reset."""
    global _phase_times
    if _phase_times:
        total = sum(dt for _, dt in _phase_times)
        parts = "  ".join(f"{n} {dt:.1f}s" for n, dt in _phase_times)
        print(f"  ⏱ {label} total {total:.1f}s  ({parts})")
    _phase_times = []


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


def _draw_voice_state(disp):
    """Mic state badge. Drawn from _draw_stats_overlay so it appears in EVERY
    window path — previously it lived only inside pump_preview(), so it
    vanished whenever the preview was drawn by OWL detection or live_preview,
    which made it look like it randomly disappeared."""
    vs = globals().get("_voice_state", "idle")
    h = disp.shape[0]
    if vs == "idle":
        col, txt = (140, 140, 140), "ready - speak a command"
    elif vs == "listening":
        col, txt = (0, 255, 0), "HEARING YOU - stop talking when done"
    else:
        col, txt = (0, 200, 255), "THINKING - please wait"
    cv2.circle(disp, (40, h - 60), 18, col, -1)
    cv2.putText(disp, txt, (70, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)


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
    _draw_voice_state(disp)

UI_LANG = "en"   # "en" or "zh" — controls all user-facing terminal text

SAY_TEXT_EN = {
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
SAY_TEXT_ZH = {
    "ok":              "好的。",
    "picking":         "正在抓取。",
    "got_it":          "抓到了。",
    "placing":         "正在放置。",
    "released":        "完成。",
    "hold_still":      "请保持不动。",
    "not_found":       "找不到目标。",
    "out_of_reach":    "够不到那里。",
    "stopping":        "停止。",
    "going_home":      "回到原点。",
    "didnt_understand":"没听懂。",
}
SAY_TEXT = SAY_TEXT_ZH if UI_LANG == "zh" else SAY_TEXT_EN

# Translation table for terminal status messages. Keys are arbitrary IDs.
# Use {placeholders} for variable interpolation.
_UI_STRINGS = {
    "looking_for":          ("  ▶ looking for '{obj}' — say 'stop' to give up",
                             "  ▶ 正在寻找 '{obj}' — 说 'stop' 取消"),
    "search_cancelled":     ("  ▶ search cancelled by user",
                             "  ▶ 搜索已取消"),
    "still_looking":        ("    still looking... ({n}/{total})",
                             "    继续寻找... ({n}/{total})"),
    "not_found_n":          ("  '{obj}' not found after {n} attempts",
                             "  '{obj}' 经过 {n} 次尝试未找到"),
    "picking_at":           ("  picking '{obj}' at ({x:.0f}, {y:.0f}) conf={s:.2f}",
                             "  抓取 '{obj}' 位置=({x:.0f}, {y:.0f}) 置信度={s:.2f}"),
    "pump_on":              ("  pump ON",
                             "  气泵已开启"),
    "pump_off":             ("  pump OFF",
                             "  气泵已关闭"),
    "pump_off_released":    ("  pump OFF — released",
                             "  气泵关闭 — 已释放"),
    "interrupted_pump_off": ("  interrupted — pump OFF",
                             "  已中断 — 关闭气泵"),
    "picked":               ("  picked",
                             "  抓取完成"),
    "chasing":              ("  ▶ chasing '{obj}'",
                             "  ▶ 正在追踪 '{obj}'"),
    "say_drop":             ("     say 'DROP' anytime to release the cube now",
                             "     随时说 'DROP' 立即释放"),
    "auto_release_hint":    ("     OR hold hand still for {s:.0f}s and it'll auto-release",
                             "     或手保持不动 {s:.0f} 秒后自动释放"),
    "placing_on":           ("  ▶ placing on '{obj}'{suffix} — waiting for it to hold still",
                             "  ▶ 放置到 '{obj}'{suffix} — 等待目标稳定"),
    "stopped_user":         ("  ▶ stopped by user",
                             "  ▶ 用户已停止"),
    "stopped_not_released": ("  ▶ stopped by user (cube NOT released)",
                             "  ▶ 用户已停止 (未释放方块)"),
    "user_release":         ("  ▶ user said release — dropping at locked position",
                             "  ▶ 收到释放指令 — 在锁定位置释放"),
    "user_release_now":     ("  ▶ user said release — forcing drop now",
                             "  ▶ 收到释放指令 — 立即释放"),
    "auto_release":         ("  ▶ auto-release after {s:.1f}s (target: {obj})",
                             "  ▶ {s:.1f} 秒后自动释放 (目标: {obj})"),
    "target_moved":         ("    target moved — re-approaching (raw={r} med={m})",
                             "    目标移动 — 重新接近 (raw={r} med={m})"),
    "in_position":          ("    in position — releasing in {s:.1f}s",
                             "    已就位 — {s:.1f} 秒后释放"),
    "target_never_settled": ("  ⚠ target never settled — keeping object",
                             "  ⚠ 目标未稳定 — 保持抓取"),
    "approach_interrupted": ("  ▶ approach interrupted — aborting place action",
                             "  ▶ 接近过程被中断 — 取消放置"),
    "release_at_hover":     ("  ▶ releasing at hover height (no descent)",
                             "  ▶ 在悬停高度释放 (不下降)"),
    "released_to_hand":     ("  pump OFF — released into hand",
                             "  气泵关闭 — 释放到手中"),
    "lowering_place":       ("  ▶ lowering to place",
                             "  ▶ 下降放置中"),
    "homing":               ("  → homing",
                             "  → 回到原点"),
    "pp_needs_both":        ("  pick_and_place needs both 'source' and 'target'",
                             "  pick_and_place 需要 'source' 和 'target'"),
    "pp_chained":           ("  ▶ chained: pick '{src}' then place on '{tgt}'",
                             "  ▶ 组合任务: 先抓取 '{src}' 再放置到 '{tgt}'"),
    "pp_pick_failed":       ("  ▶ pick failed — aborting chain (place phase skipped)",
                             "  ▶ 抓取失败 — 取消后续放置"),
    "heard":                ("\n  🎙️ heard: {text!r}",
                             "\n  🎙️ 听到: {text!r}"),
    "ignored_short":        ("  (ignored too-short utterance: {text!r})",
                             "  (忽略过短输入: {text!r})"),
    "ignored_filler":       ("  (ignored filler: {text!r})",
                             "  (忽略填充词: {text!r})"),
    "voice_input_dev":      ("  voice input: [{i}] {name}",
                             "  语音输入设备: [{i}] {name}"),
    "voice_input_default":  ("  voice input: system default (MacBook mic not found)",
                             "  语音输入: 系统默认 (未找到 MacBook 麦克风)"),
}


def t(key, **kwargs):
    """User-facing string lookup. Returns formatted string in UI_LANG."""
    en, zh = _UI_STRINGS.get(key, (key, key))
    template = zh if UI_LANG == "zh" else en
    try:
        return template.format(**kwargs)
    except Exception:
        return template


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
VAD_SILENCE_END_S = 0.6      # trailing silence to end an utterance. Short, so a
                             # command is transcribed promptly; if this is too
                             # long the listener feels "deaf" while you repeat
                             # yourself, and each repeat extends the buffer.
VAD_MIN_UTTERANCE_S = 0.25
VAD_MAX_UTTERANCE_S = 4.0    # hard cap (was 8.0 — a long ramble turned into one
                             # giant blob transcribed only after 8 seconds)
# Robustness knobs (added after mic testing — see test_mic.py). Without these,
# the stream-open transient false-triggered the gate and returned empty clips,
# and word onsets were clipped so short commands like "home" were lost.
# MEASURED: the suction pump SEALED ON A CUBE raises the mic floor ~3.7x
# (0.014 idle -> 0.053 holding on the built-in MacBook mic). With the gate at
# 0.04, 89% of frames read as "speech", so the VAD never sees trailing silence,
# the utterance never ends, and nothing is transcribed — the system goes deaf
# while holding a cube. Raise the gate whenever the pump is engaged.
# NOTE: on the BUILT-IN mic the pump (0.053) is nearly as loud as speech
# (~0.065), so no threshold fully fixes it — use the external mic.
# ADAPTIVE GATE. A FIXED pump threshold cannot work: it depends on the mic, the
# macOS input gain and whether the pump is loaded. A value tuned at input-gain
# 90 sat ABOVE the user's speech at gain 55, so the system went completely deaf
# while holding a cube. Instead we continuously estimate the noise floor from
# non-speech frames and put the gate a fixed RATIO above it — self-calibrating
# for any mic/gain, pump on or off.
VAD_NOISE_RATIO = 2.2        # gate = noise_floor * this
VAD_GATE_MIN = 0.012         # never gate below this (silence would self-trigger)
VAD_GATE_MAX = 0.090         # never gate above this (speech would be missed)
VAD_PUMP_THRESHOLD = 0.075   # legacy fixed value, kept for reference only
VAD_WARMUP_S = 0.25          # drain stream-open transient before listening
VAD_PREROLL_S = 0.30         # audio kept before trigger so onsets aren't clipped
VAD_MIN_VOICED_S = 0.20      # min loud audio to accept, else it's a false trigger

# Mic selection. None = auto (prefer names in MIC_NAME_PREFER, then any input).
# Set to an int device index (see: python test_mic.py --list) to pin one mic.
MIC_DEVICE = None
MIC_NAME_PREFER = ["外置", "external", "usb", "MacBook", "麦克风"]

voice_queue = queue.Queue()
# What the mic is doing right now, shown in the preview window and terminal so
# you can SEE whether you were heard — without it you repeat yourself, and each
# repeat extends the same utterance instead of starting a new one.
_voice_state = "idle"      # idle | listening | transcribing
_voice_stop = threading.Event()


DENOISE = True            # spectral subtraction before STT
DENOISE_ALPHA = 2.0       # over-subtraction factor (1.5-3; higher = more removal)
DENOISE_FLOOR = 0.05      # spectral floor, keeps musical noise down
DENOISE_NFFT = 512


def spectral_subtract(audio, noise, sr=SAMPLE_RATE,
                      alpha=DENOISE_ALPHA, floor=DENOISE_FLOOR):
    """Classic spectral subtraction (Boll 1979) — removes STATIONARY noise.

    The suction pump is a steady motor tone (~110Hz fundamental plus harmonics
    at 328/657/767/876Hz) sitting right in the speech band, so it can't be
    filtered out by a simple high/low-pass. But because it's stationary, its
    average magnitude spectrum can be estimated from a non-speech snippet and
    subtracted from the utterance.

      clean_mag = max(|X| - alpha*|N|,  floor*|X|)

    The floor term avoids over-subtraction artifacts ("musical noise").
    Phase is left untouched — the ear (and an ASR model) tolerate that well.

    Returns the cleaned audio, or the original if anything goes wrong.
    """
    try:
        from scipy.signal import stft, istft
        if noise is None or len(noise) < sr * 0.1 or len(audio) < sr * 0.1:
            return audio
        nper, nov = DENOISE_NFFT, DENOISE_NFFT * 3 // 4
        _, _, Z = stft(audio, sr, nperseg=nper, noverlap=nov)
        _, _, N = stft(noise, sr, nperseg=nper, noverlap=nov)
        noise_mag = np.mean(np.abs(N), axis=1, keepdims=True)
        mag, phase = np.abs(Z), np.angle(Z)
        clean = np.maximum(mag - alpha * noise_mag, floor * mag)
        _, out = istft(clean * np.exp(1j * phase), sr, nperseg=nper, noverlap=nov)
        out = np.asarray(out, dtype=np.float32)
        if not np.all(np.isfinite(out)) or len(out) < 2:
            return audio
        # keep the original peak level so VAD/STT see a familiar amplitude
        p_in, p_out = float(np.max(np.abs(audio))), float(np.max(np.abs(out)))
        if p_out > 1e-6:
            out = out * (p_in / p_out)
        return out[:len(audio)] if len(out) >= len(audio) else out
    except Exception as e:
        dlog(f"denoise failed: {e}")
        return audio


def _find_macbook_mic():
    """Pick the input device. If MIC_DEVICE is an explicit index, use it.
    Otherwise walk MIC_NAME_PREFER in order and take the first input device
    whose name contains a preferred substring; fall back to the first input
    device, then the system default."""
    try:
        devs = list(enumerate(sd.query_devices()))
    except Exception:
        return None, None

    inputs = [(i, d) for i, d in devs if d.get("max_input_channels", 0) > 0]

    # 1) explicit pin
    if isinstance(MIC_DEVICE, int):
        for i, d in inputs:
            if i == MIC_DEVICE:
                return i, d.get("name", "")

    # 2) preference order (case-insensitive substring)
    for pref in MIC_NAME_PREFER:
        p = pref.lower()
        for i, d in inputs:
            if p in d.get("name", "").lower():
                return i, d.get("name", "")

    # 3) any input device
    if inputs:
        i, d = inputs[0]
        return i, d.get("name", "")
    return None, None


def _voice_listener(stt_model):
    from collections import deque
    chunk_samples = int(SAMPLE_RATE * VAD_FRAME_S)
    silence_chunks_end = int(VAD_SILENCE_END_S / VAD_FRAME_S)
    max_chunks = int(VAD_MAX_UTTERANCE_S / VAD_FRAME_S)
    warmup_chunks = int(VAD_WARMUP_S / VAD_FRAME_S)
    preroll_chunks = int(VAD_PREROLL_S / VAD_FRAME_S)
    min_voiced = int(VAD_MIN_VOICED_S / VAD_FRAME_S)
    preroll = deque(maxlen=preroll_chunks)
    # rolling buffer of recent NON-speech audio -> the noise profile used by
    # spectral subtraction (captures whatever the pump is doing right now)
    noise_buf = deque(maxlen=int(1.5 / VAD_FRAME_S))
    noise_rms = deque(maxlen=60)   # recent non-speech levels -> adaptive gate
    buffer, silence, speaking, voiced = [], 0, False, 0
    last_heartbeat = time.time()
    device_idx, device_name = _find_macbook_mic()
    if device_idx is not None:
        print(t("voice_input_dev", i=device_idx, name=device_name))
    else:
        print(t("voice_input_default"))
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=chunk_samples, device=device_idx) as stream:
            # Drain the stream-open transient so a startup pop can't false-trigger.
            for _ in range(warmup_chunks):
                stream.read(chunk_samples)
            while not _voice_stop.is_set():
                data, _ = stream.read(chunk_samples)
                chunk = data.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                # Heartbeat every 30s so we know the listener is still alive
                if time.time() - last_heartbeat > 30:
                    last_heartbeat = time.time()
                    dlog(f"voice_listener heartbeat: rms={rms:.4f}")

                # While idle, keep a rolling pre-roll so word onsets aren't clipped.
                # Adaptive gate from the measured noise floor (see above).
                if noise_rms:
                    floor = float(np.median(noise_rms))
                    gate = min(max(floor * VAD_NOISE_RATIO, VAD_GATE_MIN),
                               VAD_GATE_MAX)
                else:
                    gate = VAD_RMS_THRESHOLD

                if not speaking and rms <= gate:
                    preroll.append(chunk)
                    noise_buf.append(chunk)
                    noise_rms.append(rms)

                if rms > gate:
                    if not speaking:
                        speaking = True
                        globals()["_voice_state"] = "listening"
                        print(f"  🎤 listening... (level {rms:.3f} > gate {gate:.3f}"
                              f"{', pump on' if _pump_engaged else ''})", flush=True)
                        buffer.extend(preroll)   # prepend pre-roll for the onset
                    silence = 0
                    voiced += 1
                    buffer.append(chunk)
                elif speaking:
                    buffer.append(chunk)         # keep trailing audio (hangover)
                    silence += 1
                    if silence >= silence_chunks_end or len(buffer) >= max_chunks:
                        audio = np.concatenate(buffer)
                        dur = len(audio) / SAMPLE_RATE
                        was_voiced = voiced
                        buffer, silence, speaking, voiced = [], 0, False, 0
                        # Discard false triggers: too little actual speech.
                        if was_voiced < min_voiced or dur < VAD_MIN_UTTERANCE_S:
                            continue
                        globals()["_voice_state"] = "transcribing"
                        print("  ⏳ transcribing...", flush=True)
                        try:
                            if DENOISE and len(noise_buf) >= 6:
                                audio = spectral_subtract(
                                    audio, np.concatenate(list(noise_buf)))
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
                                    print(t("ignored_filler", text=text))
                                    continue
                                if len(words) < 2 and len(stripped) < 5:
                                    print(t("ignored_short", text=text))
                                    continue
                            globals()["_voice_state"] = "idle"
                            text = dedupe_repeats(text)
                            print(t("heard", text=text))
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

# Short single-word commands are the ones STT mangles most (no context to lock
# onto — "home" comes back as "Oh"/"Ho"/""). Catch them (and near-misses) with a
# keyword fast-path BEFORE Qwen, so the demo doesn't depend on a clean transcript.
# TIP: in a noisy room, say the multi-word forms ("go home", "reset") — the ASR
# nails those far more reliably than the bare word.
HOME_PHRASES = {"home", "go home", "back home", "back to home", "return home",
                "reset", "reset position", "homing", "home position",
                "go to home", "return to home"}
QUIT_PHRASES = {"quit", "exit", "shut down", "shutdown"}


def _fast_command(text):
    """Map a (possibly mangled) transcript to 'home'/'quit' without the LLM.
    Returns 'home', 'quit', or None. Conservative: only fires on clear matches
    so it never hijacks a real pick/place command."""
    s = text.strip(".,!?;: ").lower()
    if s in HOME_PHRASES or s in QUIT_PHRASES:
        return "home" if s in HOME_PHRASES else "quit"
    words = s.split()
    # Single short token that clearly starts like "home" (hom / homme / homing)
    if len(words) == 1 and (words[0].startswith("hom") or words[0] == "ome"):
        return "home"
    return None


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


def owl_forward_live(model, inputs):
    """Run the OWL forward pass in a BACKGROUND thread while the main thread
    keeps the preview window repainting — so the video stays live (~30fps)
    instead of freezing for the ~500ms of inference.

    Only the worker touches torch; the main thread only does OpenCV, so there's
    no concurrent cap access or torch reentrancy.
    """
    result = {}

    def _worker():
        try:
            with torch.no_grad():
                result["out"] = model(**inputs)
        except Exception as e:
            result["err"] = e

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    while th.is_alive():
        pump_preview(0.03, label="detecting…")   # live repaint during inference
    th.join()
    if "err" in result:
        raise result["err"]
    return result["out"]


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


def _solve_ik_ikpy(target_xyz_mm, current_angles_deg, pointing_down=True):
    """ikpy iterative solver — current default."""
    target_T = np.eye(4)
    target_T[:3, 3] = np.array(target_xyz_mm) / 1000.0

    # Build initial guess, then CLAMP each entry to the URDF's joint bounds
    # so ikpy doesn't reject the seed with "Initial guess is outside of
    # provided bounds" — happens when a previous move left a joint at/past
    # its limit.
    init = [0.0] * len(chain.links)
    for i, deg in enumerate(current_angles_deg):
        init[i + 2] = float(np.radians(deg))
    for idx, link in enumerate(chain.links):
        bounds = getattr(link, "bounds", None)
        if bounds is None:
            continue
        lo, hi = bounds
        if np.isfinite(lo) and init[idx] < lo:
            init[idx] = lo + 1e-4
        if np.isfinite(hi) and init[idx] > hi:
            init[idx] = hi - 1e-4

    if pointing_down:
        target_T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        mode = "Z"
    else:
        mode = None
    try:
        j = chain.inverse_kinematics_frame(target_T, initial_position=init,
                                           orientation_mode=mode)
    except (ValueError, Exception):
        home = [0.0] * len(chain.links)
        j = chain.inverse_kinematics_frame(target_T, initial_position=home,
                                           orientation_mode=mode)
    return [float(np.degrees(j[i + 2])) for i in range(6)]


# Lazy import so ikpy-mode users don't need the ikfast module built.
_ikfast_mod = None
def _load_ikfast():
    global _ikfast_mod
    if _ikfast_mod is not None:
        return _ikfast_mod
    try:
        import ikfast_mycobot280 as _mod
        _ikfast_mod = _mod
        return _mod
    except ImportError as e:
        raise RuntimeError(
            "IK_BACKEND='ikfast' but no `ikfast_mycobot280` module found. "
            "Build it via OpenRAVE (see IKFAST_BUILD.md), place the compiled "
            "`.so`/`.pyd` next to qwen_command.py, or switch IK_BACKEND back "
            "to 'ikpy'."
        ) from e


def _solve_ik_ikfast(target_xyz_mm, current_angles_deg, pointing_down=True):
    """Analytical IK via IKFast. Returns 6 joint angles (degrees).

    IKFast can return 0..16 solutions per target — we pick the one closest to
    current joints (avoids wrist flips / big swings).
    """
    mod = _load_ikfast()
    # IKFast conventions: rotation matrix + translation in the ROBOT BASE frame.
    if pointing_down:
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    else:
        # Free orientation: default to identity; caller should specify if it matters
        R = np.eye(3)
    t_m = np.array(target_xyz_mm, dtype=np.float64) / 1000.0   # ikfast works in meters
    solutions = mod.ComputeIk(R, t_m)   # returns list of joint tuples (radians)
    if not solutions:
        raise RuntimeError(f"IKFast: no solutions for target {target_xyz_mm}")
    cur_rad = np.array([np.radians(a) for a in current_angles_deg])
    def dist(sol):
        s = np.array(sol[:6])
        return float(np.max(np.abs(s - cur_rad)))
    best = min(solutions, key=dist)
    return [float(np.degrees(a)) for a in best[:6]]


def solve_ik(target_xyz_mm, current_angles_deg, pointing_down=True):
    """Dispatch to the active IK backend."""
    if IK_BACKEND == "ikfast":
        return _solve_ik_ikfast(target_xyz_mm, current_angles_deg, pointing_down)
    return _solve_ik_ikpy(target_xyz_mm, current_angles_deg, pointing_down)


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


_affine_2x3 = None  # populated at startup if CALIB_MODE == "affine"


def pixel_to_base_at_z(u, v, mtx, dist, T_cam2base, z_target):
    if CALIB_MODE == "affine" and _affine_2x3 is not None:
        # 2D affine: pixel → base-XY at the calibrated table plane. The Z
        # argument is honored on output but the affine itself is Z-agnostic.
        # NOTE: only accurate at z ≈ 0 (the calibration plane). For thick
        # objects (cubes z=25, hand z=100), expect a few mm of error from
        # perspective foreshortening — switch back to "handeye" if that hurts.
        p = np.array([float(u), float(v), 1.0])
        xy = _affine_2x3 @ p
        return np.array([float(xy[0]), float(xy[1]), float(z_target)])
    # default: 3D back-project a ray through the pixel down to z_target plane
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
user: "pick the pink cube"               → {"action":"pick","object":"a pink cube"}
user: "pick the green cube"              → {"action":"pick","object":"a green cube"}
user: "pick the blue cube"               → {"action":"pick","object":"a blue cube"}
user: "grab the yellow one"              → {"action":"pick","object":"a yellow cube"}
user: "go to the yellow one"             → {"action":"touch","object":"a yellow cube"}
user: "put it on my hand"                → {"action":"place","object":"a human hand"}
user: "give it to me"                    → {"action":"place","object":"a human hand"}
user: "drop it in the bowl"              → {"action":"place","object":"a bowl"}
user: "place it on the yellow cube"      → {"action":"place","object":"a yellow cube"}
user: "touch the pink cube"              → {"action":"touch","object":"a pink cube"}
user: "back to home" / "reset"           → {"action":"home","object":null}
user: "stop" / "quit"                    → {"action":"quit","object":null}

pick_and_place examples (combine source AND destination in one command):
user: "hand me the green cube"           → {"action":"pick_and_place","source":"a green cube","target":"a human hand"}
user: "give me the pink one"             → {"action":"pick_and_place","source":"a pink cube","target":"a human hand"}
user: "bring me the yellow cube"         → {"action":"pick_and_place","source":"a yellow cube","target":"a human hand"}
user: "put the blue cube in the bowl"    → {"action":"pick_and_place","source":"a blue cube","target":"a bowl"}
user: "drop the green one in the box"    → {"action":"pick_and_place","source":"a green cube","target":"a cardboard box"}
user: "stack the pink on the yellow"     → {"action":"pick_and_place","source":"a pink cube","target":"a yellow cube"}
user: "pick the credit card"             → {"action":"pick","object":"a credit card"}
user: "hand me the credit card"          → {"action":"pick_and_place","source":"a credit card","target":"a human hand"}
user: "give me the bank card"            → {"action":"pick_and_place","source":"a credit card","target":"a human hand"}

Rewrite the object as a noun phrase OWL-ViT can detect. A bare color reference
("the yellow one") means a colored cube.

Color aliasing rule: the user's "purple", "magenta", or "violet" cube is
visually pink under our camera. Always normalize these colors to "pink".
"teal", "cyan", "navy" → "blue".  "lime", "olive" → "green".
  user: "pick the purple cube"   → {"action":"pick","object":"a pink cube"}
  user: "pick the cyan one"      → {"action":"pick","object":"a blue cube"}
  user: "the magenta one"        → {"action":"touch","object":"a pink cube"}

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
    for _ in range(2): cap.read()   # flush a couple stale buffered frames (was 5)
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


OWL_MIN_BBOX_AREA_PX = 1500   # reject tiny false positives (screws, dots, edges)
                              # 25mm cube at 70cm ≈ 4000 px² ; 1500 leaves margin


# HSV ranges (OpenCV Hue 0-179) for cube color refinement inside OWL's bbox.
# Same values as test_cube_hover.py.
# FALLBACK bands only — the primary path is _auto_hue_mask(), which learns the
# cube's hue from inside OWL's box. Each colour is a LIST of (lo, hi) bands.
# RED NEEDS TWO: hue is a circle 0..179 and red sits at BOTH ends. The old
# single 0..10 band missed this desk's red cube (measured hue 178.5), so the
# refinement found nothing and fell back to OWL's biased bbox centre.
CUBE_HSV_RANGES = {
    "green":  [((30, 70, 50),  (90, 255, 255))],
    "blue":   [((95, 80, 60),  (135, 255, 255))],
    "pink":   [((140, 60, 100),(172, 255, 255))],
    "yellow": [((18, 80, 80),  (35, 255, 255))],
    "red":    [((0, 90, 70),   (10, 255, 255)),      # low hue band
               ((165, 50, 70), (179, 255, 255))],    # high band (wrap-around)
}


def _color_from_query(query):
    """Extract a color name if the query mentions one — for HSV refinement."""
    q = query.lower()
    for c in CUBE_HSV_RANGES:
        if c in q:
            return c
    return None


# Words STT commonly hears instead of "cube" (from mic testing): queue, cue,
# cave, tube, koob... If the object names a known cube color and isn't a hand/
# box/card, snap it to "a <color> cube" so a misheard noun still finds the cube.
def dedupe_repeats(text):
    """Collapse a transcript that repeats the same command several times.

    Two things produce these: the user repeating because they think they were
    not heard (each repeat EXTENDS the same utterance rather than starting a
    new one), and the streaming ASR looping on noisy audio. Either way
    "Place it in a box, place it in the box, place it in the box." should
    become one command.
    """
    if not text:
        return text
    import re as _re
    parts = [p.strip() for p in _re.split(r"[.,?!]+", text) if p.strip()]
    if len(parts) < 2:
        return text
    seen, uniq = set(), []
    for p in parts:
        key = _re.sub(r"[^a-z ]", "", p.lower())
        key = " ".join(w for w in key.split() if w not in ("a", "the", "it"))
        if key and key not in seen:
            seen.add(key)
            uniq.append(p)
    if len(uniq) < len(parts):
        out = uniq[0] if len(uniq) == 1 else ". ".join(uniq)
        print(f"  (collapsed {len(parts)} repeats -> {out!r})")
        return out
    return text


def normalize_object(obj):
    if not obj:
        return obj
    o = obj.lower()
    # Never rewrite hands, containers, or flat items — those are real targets.
    if any(w in o for w in ("hand", "palm", "finger", "box", "bowl", "cup",
                            "tray", "container", "card", "coin", "paper")):
        return obj
    color = _color_from_query(o)
    if color:
        fixed = f"a {color} cube"
        if fixed != o:
            print(f"  (normalized '{obj}' → '{fixed}')")
        return fixed
    return obj


AUTO_HUE_TOL = 12       # +/- hue units kept around the learned cube hue
AUTO_HUE_MIN_SAT = 60   # ignore washed-out/grey pixels (desk, shadow, glare)
AUTO_HUE_MIN_VAL = 50


def _bands_mask(hsv, bands):
    """OR together a list of (lo, hi) HSV bands. Handles colours like red that
    need two bands because hue wraps around at 0/179."""
    if not bands:
        return None
    m = None
    for lo, hi in bands:
        # cast to uint8 — the learned hue is a float and cv2.inRange requires
        # both bounds to share the image's type
        lo_a = np.array([int(round(v)) for v in lo], dtype=np.uint8)
        hi_a = np.array([int(round(v)) for v in hi], dtype=np.uint8)
        mm = cv2.inRange(hsv, lo_a, hi_a)
        m = mm if m is None else cv2.bitwise_or(m, mm)
    return m


def _auto_hue_mask(hsv):
    """Learn the dominant hue from the CENTRE of the crop, then mask the whole
    crop with that hue +/- AUTO_HUE_TOL.

    The centre of OWL's box is almost always the object itself, so this reads
    the cube's true colour under the current lighting — no hardcoded range to
    drift out of. Wrap-around at 0/179 is handled, so red works naturally.
    """
    h, w = hsv.shape[:2]
    if h < 6 or w < 6:
        return None
    # central half of the box — least likely to contain background
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    core = hsv[cy0:cy1, cx0:cx1]
    sel = ((core[:, :, 1] >= AUTO_HUE_MIN_SAT) &
           (core[:, :, 2] >= AUTO_HUE_MIN_VAL))
    if int(sel.sum()) < 40:          # too few coloured pixels to trust
        return None
    hues = core[:, :, 0][sel].astype(float)
    # circular mean (hue is a circle 0..179)
    ang = hues * 2.0 * np.pi / 180.0
    mean_h = (np.degrees(np.arctan2(np.sin(ang).mean(),
                                    np.cos(ang).mean())) / 2.0) % 180.0
    lo_h = (mean_h - AUTO_HUE_TOL) % 180.0
    hi_h = (mean_h + AUTO_HUE_TOL) % 180.0
    s_v_lo = (AUTO_HUE_MIN_SAT, AUTO_HUE_MIN_VAL)
    if lo_h <= hi_h:
        bands = [((lo_h, s_v_lo[0], s_v_lo[1]), (hi_h, 255, 255))]
    else:   # wraps past 179 -> two bands
        bands = [((lo_h, s_v_lo[0], s_v_lo[1]), (179, 255, 255)),
                 ((0, s_v_lo[0], s_v_lo[1]), (hi_h, 255, 255))]
    return _bands_mask(hsv, bands)


def _refine_cube_center_hsv(frame, box, color, pad=6):
    """Given OWL's bbox + a color, compute the HSV centroid inside the (padded)
    bbox. Returns (cx, cy) refined pixel center, or None if HSV finds nothing.

    This is more accurate than OWL's bbox center for colored cubes because
    OWL bboxes often include shadow/edge glow that biases the center.
    """
    if color not in CUBE_HSV_RANGES:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # AUTO-HUE: learn the cube's real hue from the middle of OWL's box instead
    # of trusting a hardcoded range. Hardcoded ranges silently fail when the
    # cube's hue sits outside them (e.g. a red cube at hue 178 vs a 0..10 band),
    # and then this whole refinement returns None and we fall back to OWL's
    # biased bbox centre. Learning it also survives lighting changes.
    mask = _auto_hue_mask(hsv)
    if mask is None or int(mask.sum()) < 500:
        # fall back to the configured bands for this colour
        mask = _bands_mask(hsv, CUBE_HSV_RANGES.get(color, []))
    if mask is None or int(mask.sum()) < 500:   # not enough matching pixels
        return None
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx_local = M["m10"] / M["m00"]
    cy_local = M["m01"] / M["m00"]
    return (x1 + cx_local, y1 + cy_local)


def _filter_oversized(boxes, scores, frame_area):
    """Keep only detections whose bbox is small enough to be plausibly the
    target (not the entire desk) AND big enough to not be a random speck."""
    out = []
    for b, s in zip(boxes, scores):
        bb = b.tolist()
        area = max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])
        if area / frame_area > OWL_MAX_BBOX_AREA_FRAC:
            continue
        if area < OWL_MIN_BBOX_AREA_PX:
            continue
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
    # "cube" without a color (e.g. "the smaller cube") — still query all colors
    # so we can apply comparative filters across all detected cubes.
    if "cube" in q:
        return [f"a {col} cube" for col in CUBE_COLORS_KNOWN], None
    # Card types — credit/ID/license all look like rectangular plastic to OWL
    if "credit" in q and "card" in q:
        return CARD_QUERIES, "a credit card"
    if "id card" in q or "identification" in q:
        return CARD_QUERIES, "an ID card"
    if ("driver" in q or "driving" in q) and ("license" in q or "licence" in q):
        return CARD_QUERIES, "a driver's license"
    return [query], None


# Comparative keyword → chooser function on (box, score). Box is [x1,y1,x2,y2].
def _bbox_area(b):
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _bbox_cx(b): return (float(b[0]) + float(b[2])) / 2.0
def _bbox_cy(b): return (float(b[1]) + float(b[3])) / 2.0


COMPARATIVE_CHOOSERS = {
    # smaller / smallest → minimum bbox area
    "smaller":  ("min area",  lambda c: _bbox_area(c[0])),
    "smallest": ("min area",  lambda c: _bbox_area(c[0])),
    "small":    ("min area",  lambda c: _bbox_area(c[0])),
    # bigger / biggest / larger / largest → maximum bbox area
    "bigger":   ("max area",  lambda c: -_bbox_area(c[0])),
    "biggest":  ("max area",  lambda c: -_bbox_area(c[0])),
    "larger":   ("max area",  lambda c: -_bbox_area(c[0])),
    "largest":  ("max area",  lambda c: -_bbox_area(c[0])),
    "large":    ("max area",  lambda c: -_bbox_area(c[0])),
    "big":      ("max area",  lambda c: -_bbox_area(c[0])),
    # left / leftmost → smallest center X
    "leftmost": ("min cx",    lambda c: _bbox_cx(c[0])),
    "left":     ("min cx",    lambda c: _bbox_cx(c[0])),
    # right / rightmost → largest center X
    "rightmost":("max cx",    lambda c: -_bbox_cx(c[0])),
    "right":    ("max cx",    lambda c: -_bbox_cx(c[0])),
    # top / back-most (in image) → smallest center Y
    "top":      ("min cy",    lambda c: _bbox_cy(c[0])),
    "back":     ("min cy",    lambda c: _bbox_cy(c[0])),
    # bottom / front-most → largest center Y
    "bottom":   ("max cy",    lambda c: -_bbox_cy(c[0])),
    "front":    ("max cy",    lambda c: -_bbox_cy(c[0])),
}


def _comparative_choice(query, candidates):
    """If query contains a comparative keyword, pick the winner by that rule.
    Returns (winning_candidate, rule_name) or (None, None) if no comparative."""
    q = query.lower()
    # Word-boundary check so "left" doesn't match inside random words
    words = set(q.replace(",", " ").replace(".", " ").split())
    for kw, (rule, keyfn) in COMPARATIVE_CHOOSERS.items():
        if kw in words:
            winner = min(candidates, key=keyfn)
            dlog(f"find_object comparative: query={query!r} rule={rule} "
                 f"candidates={len(candidates)} winner_area={_bbox_area(winner[0]):.0f}")
            return winner, rule
    return None, None


def find_object(frame, query, processor, model, device, mtx, dist, T_cam2base, z_plane):
    queries, target_query = _disambiguation_queries(query)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=[queries], images=pil, return_tensors="pt").to(device)
    _t0 = time.time()
    outputs = owl_forward_live(model, inputs)
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
    # If query has a comparative keyword and we have multiple candidates, pick
    # by that rule (smaller/bigger bbox, leftmost, etc.) — otherwise default to
    # highest OWL confidence.
    if len(candidates) > 1:
        chosen, rule = _comparative_choice(query, candidates)
        if chosen is not None:
            box, score = chosen
            print(f"    (comparative pick: {rule} across {len(candidates)} candidates)")
        else:
            box, score = max(candidates, key=lambda c: c[1])
    else:
        box, score = candidates[0]
    # Default: bounding-box center.
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    # HSV refinement for colored cubes — bbox center is a few px off from the
    # true cube center; HSV centroid inside the bbox is pixel-accurate.
    color = _color_from_query(query)
    if color and "cube" in query.lower():
        refined = _refine_cube_center_hsv(frame, box, color)
        if refined is not None:
            dcx, dcy = refined[0] - cx, refined[1] - cy
            dlog(f"find_object HSV-refine: color={color} bbox_center=({cx:.1f},{cy:.1f}) → ({refined[0]:.1f},{refined[1]:.1f}) Δ=({dcx:+.1f},{dcy:+.1f})")
            cx, cy = refined
    show_preview(frame, query, box, score)
    base = pixel_to_base_at_z(cx, cy, mtx, dist, T_cam2base, z_plane)
    if base is None: return None, None
    return base[:2], score


# Firmware coordinate limits (pymycobot raises MyCobot280DataException outside
# these) and the arm's physical reach. Targets are screened against both so a
# bad detection can never crash the program.
COORD_LIMIT_MM = 281.45      # per-axis hard limit enforced by pymycobot
MAX_REACH_MM = 280.0         # spherical reach from the base origin
MIN_TIP_Z_MM = 20.0          # never drive the tip below this (table protection)


def validate_target(tx, ty, tz):
    """Return (ok, reason). Screens a tip/flange target against the firmware's
    per-axis limits, the arm's spherical reach, and a table-crash floor."""
    for name, v in (("x", tx), ("y", ty), ("z", tz)):
        if not np.isfinite(v):
            return False, f"{name}={v} is not a finite number"
        if abs(v) > COORD_LIMIT_MM:
            return False, (f"{name}={v:.0f}mm exceeds the ±{COORD_LIMIT_MM:.0f}mm "
                           f"firmware limit (bad detection?)")
    if tz < MIN_TIP_Z_MM:
        return False, f"z={tz:.0f}mm is below the {MIN_TIP_Z_MM:.0f}mm floor"
    reach = float(np.sqrt(tx * tx + ty * ty + tz * tz))
    if reach > MAX_REACH_MM:
        return False, (f"needs {reach:.0f}mm reach, arm max is {MAX_REACH_MM:.0f}mm "
                       f"(move the target closer)")
    return True, "ok"


SETTLE_TOL_MM = 1.0      # movement below this counts as "not moving"
SETTLE_STABLE_READS = 2  # consecutive stable reads before declaring settled
SETTLE_MIN_S = 0.4       # never declare settled before this (arm may not have started).
                         # 0.8 + 3 reads cost ~1.2s of pure waiting per move, and
                         # there are 3 moves per pick — that was ~3.5s of overhead.


def _send_and_wait(mc, coord_target, label):
    """One send_coords + wait for is_moving == 0. Returns (arrived_coords, err_code)."""
    import time
    try:
        mc.send_coords(coord_target, SPEED, 0)
    except Exception as e:
        # Last-resort net: validate_target should have caught this already.
        print(f"  [{label}] send_coords rejected: {e}")
        return None, None
    t0 = time.time()
    # Do NOT use is_moving() — unreliable on this arm: it reads 0 before motion
    # starts (which made a good move look like "err=365mm" because get_coords()
    # returned the stale home pose) and it flickers mid-move. Poll the actual
    # position instead and call it settled when it stops changing.
    last, stable = None, 0
    while time.time() - t0 < 15.0:
        try:
            c = mc.get_coords()
        except Exception:
            c = None
        if isinstance(c, (list, tuple)) and len(c) == 6:
            if last is not None:
                d = sum((a - b) ** 2 for a, b in zip(c[:3], last[:3])) ** 0.5
                stable = stable + 1 if d < SETTLE_TOL_MM else 0
            last = c
            if stable >= SETTLE_STABLE_READS and (time.time() - t0) >= SETTLE_MIN_S:
                break
        pump_preview(0.15, label=f"moving [{label}]")
    pump_preview(0.2, label=f"settled [{label}]")
    try: err_code = mc.get_error_information()
    except Exception: err_code = None
    return (last if last is not None else mc.get_coords()), err_code


def _move_via_native(mc, target_xyz_mm, label, require_down):
    """Motion via mycobot's firmware IK (mc.send_coords). Skips Python IK.

    Retries the same target up to 3 times if arm stops far from goal — often
    a second send from the stuck pose succeeds because joints are now in a
    different configuration.
    """
    tx, ty, tz = float(target_xyz_mm[0]), float(target_xyz_mm[1]), float(target_xyz_mm[2])
    if IK_BACKEND == "native_tool":
        # Tool frame is set at startup, so send_coords targets the PUMP TIP and
        # get_coords reports the tip. Callers pass a FLANGE target (tip + pump
        # length along Z when pointing down); recover the tip target here.
        tz = tz - PUMP_LENGTH
        coord_target = [tx, ty, tz, 180.0, 0.0, 0.0]   # matches test_cube_hover.py
    else:
        coord_target = [tx, ty, tz, 180.0, 0.0, 90.0]  # flange target, rz=90 this arm

    # Reject impossible targets BEFORE sending. pymycobot raises a hard
    # exception on out-of-range coords, which used to crash the whole program
    # when a bad detection back-projected outside the workspace.
    ok, why = validate_target(tx, ty, tz)
    if not ok:
        # If it's ONLY the height putting it out of reach (common: a target near
        # the edge with a tall hover), lower Z until it's reachable rather than
        # giving up — dropping from lower is better than not going at all.
        fixed = False
        if abs(tx) <= COORD_LIMIT_MM and abs(ty) <= COORD_LIMIT_MM:
            flat = float(np.sqrt(tx * tx + ty * ty))
            if flat < MAX_REACH_MM:
                z_max = float(np.sqrt(MAX_REACH_MM ** 2 - flat ** 2)) - 2.0
                if z_max >= MIN_TIP_Z_MM and z_max < tz:
                    print(f"  [{label}] target out of reach at z={tz:.0f}; "
                          f"lowering to z={z_max:.0f} to stay in reach")
                    tz = z_max
                    coord_target[2] = tz
                    ok, why = validate_target(tx, ty, tz)
                    fixed = ok
        if not fixed:
            print(f"  [{label}] refusing target: {why}")
            say("out_of_reach")
            return False

    MAX_ATTEMPTS = 3
    for attempt in range(1, MAX_ATTEMPTS + 1):
        actual, err_code = _send_and_wait(mc, coord_target, label)
        if not isinstance(actual, (list, tuple)) or len(actual) != 6:
            print(f"  [{label}] attempt {attempt}: couldn't read final pose")
            continue
        err = float(np.linalg.norm(np.array(actual[:3]) - np.array([tx, ty, tz])))
        tag = "" if err_code in (None, 0) else f" err_code={err_code}"
        print(f"  [{label}] attempt {attempt}: TCP=[{actual[0]:6.1f} {actual[1]:6.1f} {actual[2]:6.1f}]  "
              f"target=[{tx:6.1f} {ty:6.1f} {tz:6.1f}]  err={err:.0f}mm{tag}")
        if err <= 15:
            return True
        # else retry from this new pose

    # All attempts failed
    if require_down:
        print(f"  [{label}] {MAX_ATTEMPTS} native attempts, still {err:.0f}mm off — refusing (require_down)")
        return False
    print(f"  [{label}] {MAX_ATTEMPTS} native attempts, {err:.0f}mm off — accepting")
    return True


def move_to_tcp(mc, target_xyz_mm, label, require_down=False):
    if IK_BACKEND in ("native", "native_tool"):
        return _move_via_native(mc, target_xyz_mm, label, require_down)
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
            outputs = owl_forward_live(model, inputs)
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


PICK_SEARCH_MAX_ATTEMPTS = 60   # hard cap on frames; SEARCH_MAX_SECONDS ends it sooner
SEARCH_MAX_SECONDS = 6.0        # give up if not found within a few seconds


def do_pick(mc, cap, processor, model, device, mtx, dist, T_cam2base, obj):
    """Static-target pick: detect (retrying until found, interrupted, or
    timeout), hover, lower, pump on, lift.
    Returns True on success, False on any failure."""
    # Cards / paper / coins lie flat on the desk; need to descend almost to the
    # table surface. Cubes need to stop at the cube top.
    is_flat = any(w in obj.lower() for w in ("card", "coin", "paper", "sticker", "badge", "medal", "token"))
    z_plane = TABLE_Z_BASE_MM if is_flat else TABLE_Z_BASE_MM + CUBE_HEIGHT_MM

    # ── search loop: keep looking until OWL finds it ──
    timing_summary("(prev)")   # flush any leftover phases, start fresh for this pick
    xy, score = None, None
    print(t("looking_for", obj=obj))
    with time_phase("search"):
        search_start = time.time()
        for attempt in range(PICK_SEARCH_MAX_ATTEMPTS):
            if check_interrupt():
                print(t("search_cancelled")); say("stopping"); return False
            if time.time() - search_start > SEARCH_MAX_SECONDS:
                print(f"  gave up after {SEARCH_MAX_SECONDS:.0f}s")
                break
            frame = capture(cap)
            if frame is None: continue
            xy, score = find_object(frame, obj, processor, model, device, mtx, dist,
                                    T_cam2base, z_plane)
            if xy is not None:
                break
            if attempt > 0 and attempt % 8 == 0:
                print(t("still_looking", n=attempt, total=PICK_SEARCH_MAX_ATTEMPTS))
    if xy is None:
        print(t("not_found_n", obj=obj, n=PICK_SEARCH_MAX_ATTEMPTS))
        say("not_found")
        return False
    print(t("picking_at", obj=obj, x=xy[0], y=xy[1], s=score))
    say("picking")

    # For flat targets (card/coin/paper), descend below the assumed z=0 plane
    # since the real desk is a few mm below where calibration thinks it is.
    # For cubes, TOUCH_ABOVE=-5 presses into the rubber for a firm suction seal.
    touch = 10.0 if is_flat else TOUCH_ABOVE   # matches test_affine_landing HOVER_HEIGHT=10 → flange Z=80
    pick_tip   = np.array([xy[0], xy[1], z_plane + touch])
    hover_tip  = np.array([xy[0], xy[1], z_plane + PICK_HOVER_ABOVE])
    pick_tcp  = pick_tip + np.array([0, 0, PUMP_LENGTH])
    hover_tcp = hover_tip + np.array([0, 0, PUMP_LENGTH])

    # Move to a hover DIRECTLY above the cube (same XY, higher Z), then drop
    # straight down — no diagonal approach, no transit pose.
    with time_phase("approach"):
        ok = move_to_tcp(mc, hover_tcp, "approach")
    if not ok:
        timing_summary("pick(failed)"); return False
    # Flat targets (cards) need a perfect straight-down seal; cubes tolerate
    # a slight tilt because the rubber cup deforms around the cube top.
    with time_phase("down"):
        ok = move_to_tcp(mc, pick_tcp, "down", require_down=True)
    if not ok:
        # Failed mid-pick: arm is partly extended without a cube. Return to
        # a safer pose (the approach hover) instead of leaving it stranded.
        move_to_tcp(mc, hover_tcp, "retreat")
        timing_summary("pick(failed)"); return False
    pump_on(mc)
    print(t("pump_on"))
    say("got_it")
    with time_phase("dwell"):
        if interruptible_sleep(PUMP_DWELL_S, mc):
            print(t("interrupted_pump_off"))
            pump_off(mc); return False
    with time_phase("lift"):
        ok = move_to_tcp(mc, hover_tcp, "lift")
    if not ok:
        # interrupted/rejected during lift — keep pump engaged so the cube
        # isn't dropped on the desk
        return False
    print(t("picked"))
    timing_summary("pick")
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
        print(t("chasing", obj=obj))
        print(t("say_drop"))
        print(t("auto_release_hint", s=STABLE_HAND_SECONDS))
    else:
        print(t("placing_on", obj=obj, suffix=(' (release at hover, drop in)' if drop_in else '')))
    say("placing")

    locked_xy = None
    stable_count = 0
    recent_max_area = 0.0
    attempts = 0

    try:
        wall_start = time.time()
        while attempts < MAX_APPROACH_ATTEMPTS and (time.time() - wall_start) < MAX_APPROACH_WALL_SECONDS:
            if check_interrupt():
                _abort_and_raise(mc)
                return

            # Voice "drop" works even without a fresh OWL detection
            if locked_xy is not None and check_release_request():
                print(t("user_release"))
                hover_tcp = np.array([locked_xy[0], locked_xy[1],
                                      z_plane + hover_above + PUMP_LENGTH])
                pump_off(mc); print(t("pump_off_released")); say("released")
                time.sleep(0.4)
                move_to_tcp(mc, hover_tcp, "lift")
                return

            # Wall-time auto-release for hands and containers — works under occlusion.
            # Trust the approach: cube goes where we last detected the target.
            if drop_in and locked_xy is not None and locked_xy_time > 0:
                threshold_s = STABLE_HAND_SECONDS if is_hand else STABLE_BOX_SECONDS
                stillness_s = time.time() - locked_xy_time
                if stillness_s >= threshold_s:
                    # LAST-MOMENT GUARD: a 'wait' spoken while we were detecting
                    # lands in the queue after the top-of-loop check. Look again
                    # right before letting go, or the cube drops anyway.
                    if check_interrupt() or _wait_for_pending_speech():
                        _abort_and_raise(mc)
                        return
                    print(t("auto_release", s=stillness_s, obj=obj))
                    hover_tcp = np.array([locked_xy[0], locked_xy[1],
                                          z_plane + hover_above + PUMP_LENGTH])
                    pump_off(mc); print(t("pump_off_released")); say("released")
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
            outputs = owl_forward_live(model, inputs)
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
                    print(t("target_moved",
                            r=('yes' if moved_raw else 'no'),
                            m=('yes' if moved_med else 'no')))
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
                    print(t("approach_interrupted"))
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
                print(t("in_position", s=(STABLE_HAND_SECONDS - stillness_s)))
                pump_preview(0.4, label=f"releasing in {STABLE_HAND_SECONDS - stillness_s:.1f}s")
            else:
                show_preview(frame, obj, box, score,
                             f"stable {stable_count}/{STABLE_FRAMES_NEEDED}")
                print(f"    stable {stable_count}/{STABLE_FRAMES_NEEDED}  median@{xy.round(0)}")
            # Voice override always wins
            forced = check_release_request()
            if forced:
                print(t("user_release_now"))
            # Hand → release after STABLE_HAND_SECONDS of stillness
            # Container (box/bowl) → release after STABLE_BOX_SECONDS of stillness
            # Static cube target → release after STABLE_FRAMES_NEEDED stable frames
            ready_to_release = forced or (
                (is_hand and stillness_s >= STABLE_HAND_SECONDS)
                or (drop_in and not is_hand and stillness_s >= STABLE_BOX_SECONDS)
                or (not drop_in and stable_count >= STABLE_FRAMES_NEEDED)
            )
            if ready_to_release:
                # LAST-MOMENT GUARD (see above): re-check for a late 'wait'
                # that arrived while this frame was being detected.
                if not forced and (check_interrupt() or _wait_for_pending_speech()):
                    _abort_and_raise(mc)
                    return
                hover_tcp = np.array([xy[0], xy[1],
                                      z_plane + hover_above + PUMP_LENGTH])
                if drop_in:
                    # Hand catches the cube at hover height — no descent.
                    print(t("release_at_hover"))
                    pump_off(mc)
                    print(t("released_to_hand"))
                    time.sleep(0.6)
                    move_to_tcp(mc, hover_tcp, "lift")  # small re-affirm move
                else:
                    place_tcp = np.array([xy[0], xy[1],
                                          z_plane + TOUCH_ABOVE + PUMP_LENGTH])
                    print(t("lowering_place"))
                    if not move_to_tcp(mc, place_tcp, "down"):
                        pump_off(mc); return
                    pump_off(mc)
                    print(t("pump_off_released")); say("released")
                    time.sleep(0.4)
                    move_to_tcp(mc, hover_tcp, "lift")
                return

        print(t("target_never_settled"))
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
                outputs = owl_forward_live(model, inputs)
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
        print(t("homing")); say("going_home")
        mc.send_angles(HOME_ANGLES, SPEED); time.sleep(4)
        return
    if action == "quit":
        return
    if action == "pick_and_place":
        source = normalize_object(plan.get("source"))
        target = normalize_object(plan.get("target"))
        if not source or not target:
            print(t("pp_needs_both")); return
        print(t("pp_chained", src=source, tgt=target))
        ok = do_pick(mc, cap, processor, model, device, mtx, dist, T_cam2base, source)
        if not ok:
            print(t("pp_pick_failed"))
            return
        if check_interrupt():
            print("  ▶ interrupted between pick and place"); return
        do_place(mc, cap, processor, model, device, mtx, dist, T_cam2base, target)
        return
    obj = normalize_object(plan.get("object"))
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

SESSION_LOG = "session.log"


class _Tee:
    """Mirror everything printed to the terminal into SESSION_LOG as well, so a
    run can be reviewed (or handed to someone else) without copy-pasting the
    scrollback. Appends, so history across runs is kept."""

    def __init__(self, stream, path):
        self.stream = stream
        try:
            self.f = open(path, "a", buffering=1)
            self.f.write(f"\n===== session {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"=====\n")
        except Exception:
            self.f = None

    def write(self, s):
        self.stream.write(s)
        if self.f is not None:
            try:
                self.f.write(s)
            except Exception:
                pass

    def flush(self):
        self.stream.flush()
        if self.f is not None:
            try:
                self.f.flush()
            except Exception:
                pass

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


def main():
    global _affine_2x3
    sys.stdout = _Tee(sys.stdout, SESSION_LOG)
    print(f"(logging this session to {SESSION_LOG})")
    with open(CALIB_PATH) as f: cal = json.load(f)
    mtx = np.array(cal["camera_matrix"])
    dist = np.array(cal["dist_coeffs"])
    T_cam2base = np.array(cal["T_cam2base"])
    print(f"Calibration mode: {CALIB_MODE}")
    if CALIB_MODE == "affine":
        try:
            with open(AFFINE_CALIB_PATH) as f: aff = json.load(f)
            _affine_2x3 = np.array(aff["affine_2x3"])
            print(f"  loaded 2D affine from {AFFINE_CALIB_PATH}")
            print(f"  fit residual: mean={np.mean(aff.get('residuals_mm',[0])):.2f}mm "
                  f"max={max(aff.get('residuals_mm',[0])):.2f}mm")
        except Exception as e:
            print(f"  ⚠ failed to load affine ({e}); falling back to handeye")
            _affine_2x3 = None

    print("Loading Qwen3-1.7B...")
    llm, tok = load_llm("Qwen/Qwen3-1.7B")
    print("Loading OWL-ViT...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

    # Warm up OWL: the FIRST MPS inference compiles the graph (~seconds). Run a
    # dummy pass now so the first real pick's search isn't slow.
    try:
        print("  warming up OWL (first-inference compile)...")
        _wu = Image.fromarray(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
        _wi = processor(text=[["a cube"]], images=_wu, return_tensors="pt").to(device)
        _t = time.time()
        with torch.no_grad(): model(**_wi)
        print(f"  OWL warm ({(time.time() - _t):.1f}s)")
    except Exception as e:
        print(f"  OWL warmup skipped: {e}")

    _open_metrics_log()
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    global _live_cap
    _live_cap = cap   # let interruptible_sleep refresh the preview while we wait

    print("Connecting to robot...")
    mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
    time.sleep(2); mc.power_on(); time.sleep(0.5)

    # For the tool-frame firmware IK, tell the robot the pump is mounted so
    # send_coords drives the pump TIP (not the flange) and keeps it vertical.
    if IK_BACKEND == "native_tool":
        try:
            mc.set_tool_reference([0, 0, PUMP_LENGTH, 0, 0, 0])
            mc.set_end_type(1)   # 1 = tool frame
            print(f"IK: native_tool — pump tip = flange + {PUMP_LENGTH:.0f}mm")
        except Exception as e:
            print(f"⚠ could not set tool frame ({e}); firmware IK won't know the pump")

    print("Homing arm to [0,0,0,0,0,0]...")
    mc.send_angles(HOME_ANGLES, SPEED)
    time.sleep(4)

    if INPUT_MODE == "voice":
        print("Loading STT (Nemotron)...")
        STT_MODEL = "nemotron"   # ← change to "parakeet" to try the newer model
        STT_OPTIONS = {
            "nemotron": "mlx-community/nemotron-3.5-asr-streaming-0.6b",
            "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
        }
        print(f"Loading STT ({STT_MODEL}): {STT_OPTIONS[STT_MODEL]}")
        stt = load_stt(STT_OPTIONS[STT_MODEL])
        start_voice(stt)
        print("  ▶ mic is live — speak a command\n")
    else:
        print("  ▶ text-only mode — type commands at the prompt\n")

    print(f"Ready. Mode: {INPUT_MODE}. Ctrl+C to exit.\n")
    if INPUT_MODE == "text":
        print("cmd> ", end="", flush=True)
    try:
        while True:
            if INPUT_MODE == "text":
                # Non-blocking stdin poll — keeps the OpenCV window responsive.
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    text = sys.stdin.readline()
                    if not text:  # EOF
                        break
                else:
                    pump_preview(0.10, label="text-mode idle")
                    continue
            else:
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

            # Bare release words at the prompt: if pump is holding something,
            # release it in place. Otherwise say "nothing to drop" — DON'T let
            # Qwen turn "drop it" into a new place action.
            stripped_words = set(text.lower().replace(",", " ").replace(".", " ").split())
            if stripped_words and stripped_words.issubset(
                    RELEASE_WORDS | {"it", "the", "cube", "please"}):
                if _pump_engaged:
                    print("  ▶ releasing at current position")
                    pump_off(mc); say("released")
                else:
                    print("  ▶ nothing to drop — pump isn't holding anything")
                if INPUT_MODE == "text":
                    print("cmd> ", end="", flush=True)
                continue

            # Fast-path short commands the LLM+STT combo tends to fumble.
            fast = _fast_command(text)
            if fast == "quit":
                break
            if fast == "home":
                plan = {"action": "home", "object": None}
            else:
                try:
                    with time_phase("qwen_plan"):
                        plan = qwen_plan(llm, tok, text)
                except Exception as e:
                    print(f"  ⚠ plan parse failed: {e}"); say("didnt_understand")
                    continue
            # NEVER let the LLM quit on a garbled transcript. Nemotron heard
            # "give me" as "V me", Qwen turned that into {"action":"quit"} and
            # the whole session exited mid-demo. Quitting now requires an
            # explicit phrase (QUIT_PHRASES, matched by _fast_command above).
            if plan.get("action") == "quit" and fast != "quit":
                print(f"  ⚠ ignoring 'quit' inferred from {text!r} — say "
                      f"'quit' or 'exit' explicitly, or press Ctrl+C")
                say("didnt_understand")
                if INPUT_MODE == "text":
                    print("cmd> ", end="", flush=True)
                continue
            print(f"  plan: {plan}"); say("ok")
            if plan["action"] == "quit":
                break
            execute(plan, mc, cap, processor, model, device, mtx, dist, T_cam2base)
            if INPUT_MODE == "text":
                print("cmd> ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        _voice_stop.set()
        try: pump_off(mc)
        except Exception: pass
        if IK_BACKEND == "native_tool":
            try: mc.set_end_type(0)   # restore flange frame for other scripts
            except Exception: pass
        cap.release()
        print("Done.")


if __name__ == "__main__":
    main()
