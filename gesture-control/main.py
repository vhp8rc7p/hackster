"""
Temporal Gesture Recognition System + MyCobot280 Control
Detects: Horizontal Wipes, Vertical Lifts/Drops, Z-Axis Pushes/Pulls

Architecture:
  Layer A: Data Acquisition       (OpenCV VideoCapture)
  Layer B: Landmark Inference     (MediaPipe Hands)
  Layer C: Feature Engineering    (NumPy - palm center, normal, scale)
  Layer D: Temporal Buffer        (collections.deque)
  Engine:  Heuristic State Machine (threshold-based gesture logic)
  Layer E: Robot Controller       (pymycobot MyCobot280, background thread)
"""

import sys
import cv2
import mediapipe as mp

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import time
import threading
import urllib.request
import os

# ── Detect which MediaPipe API generation is available
try:
    from mediapipe.tasks import python as _mp_tasks_probe  # noqa
    _USE_TASKS_API = True
except ImportError:
    _USE_TASKS_API = False


# ─────────────────────────────────────────────
# 1. CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────

BUFFER_SIZE       = 20       # frames held in the temporal buffer
FPS_TARGET        = 30

# Gesture thresholds (tune to your environment)
WIPE_X_THRESHOLD  = 0.12     # normalized ΔX across the window
WIPE_Y_VAR_MAX    = 0.015    # variance of Y must stay low for a clean wipe
LIFT_Y_THRESHOLD  = 0.18     # normalized ΔY across the window
LIFT_NORMAL_DOT   = 0.6      # palm must face roughly toward camera
PUSH_SCALE_RATIO  = 0.22     # ≥22 % change in hand scale

COOLDOWN_SECONDS  = 0.8      # minimum time between two gesture firings

# MediaPipe landmark indices
WRIST             = 0
INDEX_MCP         = 5
MIDDLE_MCP        = 9
PINKY_MCP         = 17
RING_MCP          = 13

PALM_LANDMARKS    = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Robot configuration
ROBOT_PORT        = "/dev/tty.usbserial-59010016581"
ROBOT_BAUD        = 115200
ROBOT_STEP_MM     = 20.0     # mm moved per gesture
ROBOT_WIPE_STEP_MM = 40.0    # mm moved per left/right swipe
ROBOT_SPEED       = 80       # % of max speed (0-100)

# Safety workspace bounds (mm) — adjust to your robot's actual range
ROBOT_X_BOUNDS    = (-300.0,  300.0)
ROBOT_Y_BOUNDS    = (-300.0,  300.0)
ROBOT_Z_BOUNDS    = (  50.0,  350.0)


# ─────────────────────────────────────────────
# 2. DATA TYPES
# ─────────────────────────────────────────────

class Gesture(Enum):
    NONE        = auto()
    WIPE_RIGHT  = auto()
    WIPE_LEFT   = auto()
    LIFT_UP     = auto()
    DROP_DOWN   = auto()
    PUSH_IN     = auto()
    PULL_OUT    = auto()

    def label(self) -> str:
        return {
            Gesture.NONE:       "–",
            Gesture.WIPE_RIGHT: "→  WIPE RIGHT",
            Gesture.WIPE_LEFT:  "←  WIPE LEFT",
            Gesture.LIFT_UP:    "↑  LIFT UP",
            Gesture.DROP_DOWN:  "↓  DROP DOWN",
            Gesture.PUSH_IN:    "⊙  PUSH IN",
            Gesture.PULL_OUT:   "⊗  PULL OUT",
        }[self]

    def color(self):
        return {
            Gesture.NONE:       (180, 180, 180),
            Gesture.WIPE_RIGHT: ( 50, 220, 100),
            Gesture.WIPE_LEFT:  ( 50, 220, 100),
            Gesture.LIFT_UP:    ( 80, 180, 255),
            Gesture.DROP_DOWN:  ( 80, 180, 255),
            Gesture.PUSH_IN:    (255, 180,  50),
            Gesture.PULL_OUT:   (255, 180,  50),
        }[self]


@dataclass
class FrameFeatures:
    """Extracted per-frame features stored in the temporal buffer."""
    palm_center: np.ndarray   # shape (3,) – (x, y, z) normalized
    palm_normal: np.ndarray   # shape (3,) – unit normal of the palm plane
    hand_scale:  float        # wrist-to-middle-MCP distance (normalized)
    timestamp:   float        # time.time()


# ─────────────────────────────────────────────
# 3. LAYER B – LANDMARK INFERENCE WRAPPER
# ─────────────────────────────────────────────

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"[MediaPipe] Downloading hand landmarker model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[MediaPipe] Download complete.")


class _LandmarkProxy:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class HandLandmarkEngine:
    """
    Unified wrapper that works with both MediaPipe API generations:
      • Legacy  (< 0.10)  – mp.solutions.hands
      • Modern  (≥ 0.10)  – mediapipe.tasks.vision.HandLandmarker
    """

    def __init__(self, max_hands: int = 1, min_detection_conf: float = 0.7,
                 min_tracking_conf: float = 0.6):
        if _USE_TASKS_API:
            self._init_tasks(max_hands, min_detection_conf, min_tracking_conf)
        else:
            self._init_solutions(max_hands, min_detection_conf, min_tracking_conf)

    def _init_tasks(self, max_hands, det_conf, track_conf):
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        _ensure_model()

        base_opts = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._mode       = "tasks"

        self._connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        ]

    def _init_solutions(self, max_hands, det_conf, track_conf):
        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self._mp_draw      = mp.solutions.drawing_utils
        self._draw_spec_lm = mp.solutions.drawing_utils.DrawingSpec(
            color=(255, 255, 255), thickness=1, circle_radius=3)
        self._draw_spec_cn = mp.solutions.drawing_utils.DrawingSpec(
            color=(100, 200, 255), thickness=2)
        self._mode = "solutions"

    def process(self, bgr_frame: np.ndarray):
        """
        Returns:
            landmarks (list of objects with .x/.y/.z) or None
            annotated_frame (np.ndarray)
        """
        if self._mode == "tasks":
            return self._process_tasks(bgr_frame)
        return self._process_solutions(bgr_frame)

    def _process_tasks(self, bgr_frame: np.ndarray):
        import mediapipe as _mp
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)

        ts_ms  = int(time.time() * 1000)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        annotated = bgr_frame.copy()
        if not result.hand_landmarks:
            return None, annotated

        # Prefer the hand labelled "Left" by MediaPipe (= user's physical right
        # hand in a mirrored frame). Fall back to the first detected hand.
        target_idx = 0
        for i, handedness in enumerate(result.handedness):
            if handedness[0].category_name == "Left":
                target_idx = i
                break

        raw_lms = result.hand_landmarks[target_idx]
        proxies = [_LandmarkProxy(lm.x, lm.y, lm.z) for lm in raw_lms]
        h, w = annotated.shape[:2]
        pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in proxies]
        for a, b in self._connections:
            cv2.line(annotated, pts_px[a], pts_px[b], (100, 200, 255), 2)
        for px, py in pts_px:
            cv2.circle(annotated, (px, py), 3, (255, 255, 255), -1)
        return proxies, annotated

    def _process_solutions(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True
        annotated = bgr_frame.copy()

        if results.multi_hand_landmarks:
            # Prefer "Left" label (= user's physical right hand in mirrored frame).
            # Fall back to first detected hand if right hand not visible.
            target_idx = 0
            for i, handedness in enumerate(results.multi_handedness):
                if handedness.classification[0].label == "Left":
                    target_idx = i
                    break
            lms = results.multi_hand_landmarks[target_idx]
            self._mp_draw.draw_landmarks(
                annotated, lms,
                self._mp_hands.HAND_CONNECTIONS,
                self._draw_spec_lm,
                self._draw_spec_cn,
            )
            return list(lms.landmark), annotated

        return None, annotated

    def close(self):
        if self._mode == "tasks":
            self._landmarker.close()
        else:
            self._hands.close()


# ─────────────────────────────────────────────
# 4. LAYER C – FEATURE ENGINEERING
# ─────────────────────────────────────────────

def extract_features(landmarks) -> FrameFeatures:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # (21, 3)

    palm_center = pts[PALM_LANDMARKS].mean(axis=0)

    wrist_pt  = pts[WRIST]
    index_vec = pts[INDEX_MCP] - wrist_pt
    pinky_vec = pts[PINKY_MCP] - wrist_pt
    normal    = np.cross(index_vec, pinky_vec)
    norm_mag  = np.linalg.norm(normal)
    if norm_mag > 1e-6:
        normal /= norm_mag

    hand_scale = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))

    return FrameFeatures(
        palm_center=palm_center,
        palm_normal=normal,
        hand_scale=hand_scale,
        timestamp=time.time(),
    )


def is_fist(landmarks) -> bool:
    """
    True when 3+ fingers are curled.
    Uses 3-D tip-to-wrist distance vs MCP-to-wrist distance so a hand
    pointing downward (during a DROP gesture) doesn't trigger false positives.
    A curled finger has its tip closer to the wrist than the MCP knuckle is.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = pts[0]
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    curled = 0
    for tip_i, mcp_i in zip(tips, mcps):
        tip_dist = np.linalg.norm(pts[tip_i] - wrist)
        mcp_dist = np.linalg.norm(pts[mcp_i] - wrist)
        if mcp_dist > 1e-6 and tip_dist < mcp_dist * 1.2:
            curled += 1
    return curled >= 3


# ─────────────────────────────────────────────
# 5. LAYER D – TEMPORAL BUFFER
# ─────────────────────────────────────────────

class TemporalBuffer:
    """Circular buffer storing the last N FrameFeatures objects."""

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._buf: deque[FrameFeatures] = deque(maxlen=maxlen)

    def push(self, feat: FrameFeatures):
        self._buf.append(feat)

    def full(self) -> bool:
        return len(self._buf) == self._buf.maxlen

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def maxlen(self) -> int:
        return self._buf.maxlen  # type: ignore[return-value]

    def centers(self) -> np.ndarray:
        return np.array([f.palm_center for f in self._buf])

    def normals(self) -> np.ndarray:
        return np.array([f.palm_normal for f in self._buf])

    def scales(self) -> np.ndarray:
        return np.array([f.hand_scale for f in self._buf])

    def clear(self):
        self._buf.clear()


# ─────────────────────────────────────────────
# 6. GESTURE LOGIC ENGINE (Threshold State Machine)
# ─────────────────────────────────────────────

_UP_VEC  = np.array([0.0, -1.0, 0.0])   # screen Y is inverted: up = -Y
_CAM_VEC = np.array([0.0,  0.0, -1.0]) # palm facing camera = negative Z


class GestureEngine:
    def __init__(self):
        self._last_fire_time: float = 0.0

    def _cooled_down(self) -> bool:
        return (time.time() - self._last_fire_time) >= COOLDOWN_SECONDS

    def _fire(self, gesture: Gesture) -> Gesture:
        self._last_fire_time = time.time()
        return gesture

    def evaluate(self, buf: TemporalBuffer) -> Gesture:
        if not buf.full() or not self._cooled_down():
            return Gesture.NONE

        centers = buf.centers()
        normals = buf.normals()
        scales  = buf.scales()

        delta = centers[-1] - centers[0]
        dx    = delta[0]
        dy    = delta[1]

        # ── Wipe ────────────────────────────────────────────────────────
        y_variance = float(np.var(centers[:, 1]))
        if abs(dx) > WIPE_X_THRESHOLD and y_variance < WIPE_Y_VAR_MAX:
            return self._fire(Gesture.WIPE_RIGHT if dx > 0 else Gesture.WIPE_LEFT)

        # ── Lift / Drop ──────────────────────────────────────────────────
        avg_normal = normals.mean(axis=0)
        norm_mag   = np.linalg.norm(avg_normal)
        if norm_mag > 1e-6:
            avg_normal /= norm_mag
        dot_cam = float(np.dot(avg_normal, _CAM_VEC))

        if abs(dy) > LIFT_Y_THRESHOLD and abs(dot_cam) > LIFT_NORMAL_DOT:
            return self._fire(Gesture.LIFT_UP if dy < 0 else Gesture.DROP_DOWN)

        # ── Push / Pull ──────────────────────────────────────────────────
        # Only fire if hand isn't also moving sideways or vertically
        scale_start = float(scales[:5].mean())
        scale_end   = float(scales[-5:].mean())
        if scale_start > 1e-6 and abs(dx) < WIPE_X_THRESHOLD and abs(dy) < 0.30:
            scale_ratio = abs(scale_end - scale_start) / scale_start
            if scale_ratio > PUSH_SCALE_RATIO:
                return self._fire(
                    Gesture.PUSH_IN if scale_end < scale_start else Gesture.PULL_OUT
                )

        return Gesture.NONE


# ─────────────────────────────────────────────
# 7. OVERLAY RENDERER
# ─────────────────────────────────────────────

class OverlayRenderer:
    def __init__(self):
        self._last_gesture  = Gesture.NONE
        self._gesture_time  = 0.0
        self._display_secs  = 1.5

    def update_gesture(self, gesture: Gesture):
        if gesture != Gesture.NONE:
            self._last_gesture = gesture
            self._gesture_time = time.time()

    def draw(self, frame: np.ndarray, buf: TemporalBuffer,
             features: FrameFeatures | None, fps: float,
             robot_coords: list | None = None,
             paused: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, 48), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        if paused:
            cv2.putText(frame, "|| PAUSED (FIST)", (w // 2 - 130, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2, cv2.LINE_AA)

        if buf.full():
            bar_color = (80, 220, 120)
            buf_text  = "BUFFER FULL"
        else:
            ratio     = len(buf) / buf.maxlen
            bar_color = (100, 100, 220)
            buf_text  = f"BUFFERING {int(ratio * 100):3d}%"
        cv2.putText(frame, buf_text, (w - 200, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_color, 1, cv2.LINE_AA)

        if features is not None:
            pc = features.palm_center
            pn = features.palm_normal
            cv2.putText(frame,
                f"Center: ({pc[0]:.2f}, {pc[1]:.2f}, {pc[2]:.3f})",
                (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame,
                f"Normal: ({pn[0]:.2f}, {pn[1]:.2f}, {pn[2]:.2f})",
                (10, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame,
                f"Scale:  {features.hand_scale:.4f}",
                (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)

        # Robot coordinate readout (bottom-right)
        if robot_coords is not None:
            rob_text = (f"Robot  X:{robot_coords[0]:7.1f}  "
                        f"Y:{robot_coords[1]:7.1f}  "
                        f"Z:{robot_coords[2]:7.1f}")
            (tw, _), _ = cv2.getTextSize(rob_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.putText(frame, rob_text, (w - tw - 10, h - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 230, 255), 1, cv2.LINE_AA)

        elapsed = time.time() - self._gesture_time
        if self._last_gesture != Gesture.NONE and elapsed < self._display_secs:
            alpha  = max(0.0, 1.0 - elapsed / self._display_secs)
            color  = tuple(int(c * alpha) for c in self._last_gesture.color())
            label  = self._last_gesture.label()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)
            tx = (w - tw) // 2
            ty = h // 2
            cv2.putText(frame, label, (tx + 2, ty + 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2, cv2.LINE_AA)

        return frame


# ─────────────────────────────────────────────
# 8. ROBOT CONTROLLER
# ─────────────────────────────────────────────

# Gesture → (axis_index, sign)   axis: 0=X  1=Y  2=Z
_GESTURE_AXIS = {
    Gesture.WIPE_RIGHT: (1, -1),
    Gesture.WIPE_LEFT:  (1, +1),
    Gesture.LIFT_UP:    (2, +1),
    Gesture.DROP_DOWN:  (2, -1),
    Gesture.PUSH_IN:    (0, -1),
    Gesture.PULL_OUT:   (0, +1),
}
_AXIS_BOUNDS = [ROBOT_X_BOUNDS, ROBOT_Y_BOUNDS, ROBOT_Z_BOUNDS]


class RobotController:
    """
    Tracks position internally — never calls get_coords() at runtime.
    Gets initial position once at startup then maintains it locally.
    """

    def __init__(self, port: str = ROBOT_PORT, baud: int = ROBOT_BAUD,
                 step_mm: float = ROBOT_STEP_MM):
        self._port = port
        self._baud = baud
        self._step = step_mm
        self._mc   = self._open()

        # Read position once at startup; fall back to a safe default
        self._coords = self._read_initial_coords()
        print(f"[Robot] Starting coords: {self._coords}")

        self._pending: Gesture | None = None
        self._lock   = threading.Lock()
        self._event  = threading.Event()

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"[Robot] Connected on {port} @ {baud} baud.")

    def _open(self):
        from pymycobot import MyPalletizer260
        mc = MyPalletizer260(self._port, self._baud)
        time.sleep(1.5)
        print("[Robot] Waiting for arm firmware to be ready...")
        angles = None
        for _ in range(20):
            angles = mc.get_angles()
            if isinstance(angles, list):
                break
            time.sleep(0.3)
        # Check if coords already work (arm was previously commanded)
        mc._serial_port.reset_input_buffer()
        test = mc.get_coords()
        if not isinstance(test, list) or len(test) < 3:
            # Arm hasn't moved yet — go_home() forces FK recompute
            print("[Robot] Moving to home to activate coordinate mode...")
            mc.go_home()
            time.sleep(3.0)
        print("[Robot] Arm ready.")
        return mc

    def _read_initial_coords(self) -> list:
        for _ in range(20):
            self._mc._serial_port.reset_input_buffer()
            c = self._mc.get_coords()
            if isinstance(c, list) and len(c) >= 3:
                print(f"[Robot] Got real coords: {c}")
                return c[:]
            time.sleep(0.3)
        print("[Robot] WARNING: using fallback coords — arm position unknown")
        return [160.0, 0.0, 220.0, 0.0]

    def _reconnect(self):
        print("[Robot] Port lost — reconnecting...")
        try:
            self._mc.close()
        except Exception:
            pass
        time.sleep(1.5)
        self._mc = self._open()
        self._coords = self._read_initial_coords()
        print("[Robot] Reconnected.")

    # ── Background worker ────────────────────────────────────────────────
    def _worker(self):
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                gesture = self._pending
                self._pending = None
            if gesture is None:
                break
            try:
                self._execute(gesture)
            except PermissionError:
                self._reconnect()
            except Exception as exc:
                print(f"[Robot] ERROR in _execute: {exc}")

    def _execute(self, gesture: Gesture):
        if gesture not in _GESTURE_AXIS:
            return

        axis, sign = _GESTURE_AXIS[gesture]
        lo, hi = _AXIS_BOUNDS[axis]
        step = ROBOT_WIPE_STEP_MM if gesture in (Gesture.WIPE_LEFT, Gesture.WIPE_RIGHT) else self._step

        with self._lock:
            coords = self._coords[:]
        coords[axis] = float(np.clip(coords[axis] + sign * step, lo, hi))

        self._mc.send_coords(coords, ROBOT_SPEED)
        time.sleep(1.5)

        with self._lock:
            self._coords = coords[:]
        print(f"[Robot] {gesture.label()}  X:{coords[0]:.1f}  Y:{coords[1]:.1f}  Z:{coords[2]:.1f}")

    # ── Public API ───────────────────────────────────────────────────────
    def apply_gesture(self, gesture: Gesture):
        with self._lock:
            self._pending = gesture
        self._event.set()

    @property
    def last_coords(self) -> list | None:
        with self._lock:
            return self._coords[:]

    def close(self):
        with self._lock:
            self._pending = None
        self._event.set()
        self._thread.join(timeout=2)
        try:
            self._mc.close()
        except Exception:
            pass


# ─────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────

def run(camera_index: int = 0, width: int = 1280, height: int = 720,
        robot_port: str = ROBOT_PORT, robot_baud: int = ROBOT_BAUD,
        robot_step_mm: float = ROBOT_STEP_MM, connect_robot: bool = True):
    """
    Main entry point.
    Keys:  Q = quit   R = reset buffer
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)

    engine   = HandLandmarkEngine()
    buffer   = TemporalBuffer(maxlen=BUFFER_SIZE)
    gesture  = GestureEngine()
    renderer = OverlayRenderer()

    robot: RobotController | None = None
    if connect_robot:
        try:
            robot = RobotController(robot_port, robot_baud, robot_step_mm)
        except Exception as exc:
            print(f"[Robot] Connection failed: {exc}")
            print("[Robot] Running in gesture-only mode.")

    prev_time   = time.time()
    paused      = False   # True while fist is held
    print("[GestureRecognizer] Running — press Q to quit, R to reset buffer.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[GestureRecognizer] Camera feed lost.")
                break

            frame = cv2.flip(frame, 1)

            landmarks, annotated = engine.process(frame)

            current_features = None
            detected         = Gesture.NONE

            if landmarks:
                if is_fist(landmarks):
                    paused = True
                    buffer.clear()
                else:
                    paused = False
                    current_features = extract_features(landmarks)
                    buffer.push(current_features)
                    detected = gesture.evaluate(buffer)
                    if detected != Gesture.NONE:
                        print(f"[Gesture] {detected.label()}")
                        buffer.clear()
                        if robot is not None:
                            robot.apply_gesture(detected)
            else:
                paused = False

            renderer.update_gesture(detected)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-9)
            prev_time = now

            robot_coords = robot.last_coords if robot else None
            output = renderer.draw(annotated, buffer, current_features, fps, robot_coords, paused)
            cv2.imshow("Gesture Recognizer", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                buffer.clear()
                print("[GestureRecognizer] Buffer reset.")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        engine.close()
        if robot is not None:
            robot.close()
        cv2.destroyAllWindows()
        print("[GestureRecognizer] Exited cleanly.")


# ─────────────────────────────────────────────
# 10. CALLBACK / API  (for external integration)
# ─────────────────────────────────────────────

class GestureRecognizer:
    """
    Headless API for embedding in robotics or larger applications.

    Usage:
        recognizer = GestureRecognizer(callback=my_handler)
        for frame in my_source:
            recognizer.feed(frame)

    The callback receives a Gesture enum value each time one fires.
    """

    def __init__(self, callback=None, buffer_size: int = BUFFER_SIZE):
        self._engine   = HandLandmarkEngine()
        self._buffer   = TemporalBuffer(maxlen=buffer_size)
        self._gesture  = GestureEngine()
        self._callback = callback

    def feed(self, bgr_frame: np.ndarray) -> Gesture:
        landmarks, _ = self._engine.process(bgr_frame)
        if landmarks:
            features = extract_features(landmarks)
            self._buffer.push(features)
            result = self._gesture.evaluate(self._buffer)
            if result != Gesture.NONE and self._callback:
                self._callback(result)
            return result
        return Gesture.NONE

    def reset(self):
        self._buffer.clear()

    def close(self):
        self._engine.close()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Gesture-controlled MyCobot280")
    ap.add_argument("--camera",   type=int,   default=0,              help="Camera index")
    ap.add_argument("--port",     type=str,   default=ROBOT_PORT,     help="Serial port (e.g. COM7)")
    ap.add_argument("--baud",     type=int,   default=ROBOT_BAUD,     help="Baud rate")
    ap.add_argument("--step",     type=float, default=ROBOT_STEP_MM,  help="mm per gesture")
    ap.add_argument("--no-robot", action="store_true",                help="Run without robot")
    args = ap.parse_args()

    run(
        camera_index  = args.camera,
        robot_port    = args.port,
        robot_baud    = args.baud,
        robot_step_mm = args.step,
        connect_robot = not args.no_robot,
    )
