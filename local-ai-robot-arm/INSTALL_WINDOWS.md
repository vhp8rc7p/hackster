# Windows Installation Guide (with NVIDIA GPU)

Getting the voice-controlled robot arm demo running on a Windows PC with
NVIDIA GPU. **Not a drop-in port** — the Mac stack uses MLX (Apple Silicon
only), so we swap those two components for Windows-friendly equivalents.

## What changes vs the Mac version

| Component | Mac (MLX) | Windows (this guide) |
|---|---|---|
| **LLM** (command parsing) | `mlx-lm` + Qwen3-1.7B | **Ollama** + Qwen3-1.7B |
| **STT** (speech-to-text) | `mlx-audio` + Nemotron | **Whisper** (`openai-whisper`) |
| **Object detection** | OWLv2 (PyTorch + MPS) | OWLv2 (**PyTorch + CUDA**) |
| **Robot** | `pymycobot` + `/dev/tty.usbserial-*` | `pymycobot` + `COMx` |
| **TTS** | macOS `say` + prebuilt WAVs | prebuilt WAVs only, or `pyttsx3` |

Result: same functionality, different backends. The `qwen_command.py`
script needs 3 imports swapped (see "Code changes" section at the end).

---

## 1. Prerequisites

Install these first (~10 minutes):

- **Python 3.11 (64-bit)** from https://python.org — check "Add Python to PATH"
- **Git for Windows** from https://git-scm.com
- **Ollama for Windows** from https://ollama.com/download — installs as a background service
- **NVIDIA driver** — latest from https://nvidia.com/download (check with `nvidia-smi` in cmd)
- **CUDA Toolkit 12.1** (matches PyTorch build below) — https://developer.nvidia.com/cuda-downloads

Verify each in a fresh cmd window:
```cmd
python --version    :: 3.11.x
git --version
ollama --version
nvidia-smi          :: should show your GPU + CUDA version
```

---

## 2. Clone the repo + venv

```cmd
cd C:\Users\%USERNAME%\
git clone https://github.com/vhp8rc7p/hackster.git
cd hackster\local-ai-robot-arm
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## 3. Install PyTorch with CUDA

Critical: install PyTorch **BEFORE** other packages, and use the CUDA build:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is working:
```cmd
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```
Should print `cuda: True <YourGPU>`.

---

## 4. Install everything else

```cmd
pip install transformers pillow numpy opencv-python
pip install openai-whisper
pip install sounddevice soundfile
pip install pymycobot
pip install ikpy scipy
pip install requests   :: for Ollama API calls
```

---

## 5. Pull the LLM via Ollama

```cmd
ollama pull qwen3:1.7b
```

Downloads ~1.4 GB. Test it:
```cmd
ollama run qwen3:1.7b "Say hi in one word"
```

Ollama runs as a background service and exposes an HTTP API at
`http://localhost:11434`.

---

## 6. Download Whisper model

The `openai-whisper` package downloads models on first use. To pre-download
the small English model (~250 MB, plenty for command parsing):

```cmd
python -c "import whisper; whisper.load_model('small.en')"
```

Larger models available: `medium.en` (~1.5 GB), `large-v3` (~3 GB). Small
is fine for short voice commands.

---

## 7. Robot serial port (Windows convention)

On Windows, mycobot shows up as `COM3`, `COM4`, etc. Find it:

1. Plug the mycobot into USB
2. Open **Device Manager** → Ports (COM & LPT)
3. Look for "USB Serial Port (COMx)" — note the COM number

You'll set this in the Python scripts as e.g. `SERIAL_PORT = "COM3"`.

Test:
```cmd
python -c "from pymycobot.mycobot280 import MyCobot280; import time; mc=MyCobot280('COM3', 115200); time.sleep(2); print(mc.get_angles())"
```

---

## 8. Camera

Windows uses DirectShow. In OpenCV code, force the backend:

```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

Camera IDs on Windows: 0 = first camera found (usually built-in webcam),
1 = second, etc. Trial and error if you have multiple.

Grant Python camera permission: **Settings → Privacy & security → Camera →
enable "Let desktop apps access your camera"**.

---

## 9. Code changes needed

The three main scripts (`qwen_command.py`, calibration scripts) have
Mac-specific imports and paths. Minimum changes for Windows:

**Serial port** — everywhere `SERIAL_PORT = "/dev/tty.usbserial-*"`:
```python
SERIAL_PORT = "COM3"    # your COM number
```

**Camera backend** — everywhere `cv2.VideoCapture(CAMERA_ID)`:
```python
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
```

**LLM (Ollama replaces `mlx_lm`)** — in `qwen_command.py`, replace the
`load_llm` / `generate_llm` calls with:
```python
import requests
def qwen_generate(prompt, system=None):
    payload = {
        "model": "qwen3:1.7b",
        "prompt": prompt,
        "system": system or "",
        "stream": False,
        "options": {"num_predict": 200},
    }
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
    return r.json()["response"]
```

**STT (Whisper replaces `mlx_audio.stt`)**:
```python
import whisper
stt_model = whisper.load_model("small.en")
result = stt_model.transcribe("captured.wav", language="en", fp16=True)
text = result["text"]
```

**TTS (drop `say`, use prebuilt WAVs)** — already handled if
`SPEECH_ENABLED = True` and TTS wavs are in `tts/`. Or use `pyttsx3` for
live synthesis (`pip install pyttsx3`).

---

## 10. Smoke test

After all above:

```cmd
:: Verify robot connection
python -c "from pymycobot.mycobot280 import MyCobot280; import time; mc=MyCobot280('COM3',115200); time.sleep(2); print('angles:', mc.get_angles())"

:: Verify camera
python -c "import cv2; cap=cv2.VideoCapture(0, cv2.CAP_DSHOW); ok,f=cap.read(); print('camera:', 'OK' if ok else 'FAILED', f.shape if ok else '')"

:: Verify Ollama
python -c "import requests; r=requests.post('http://localhost:11434/api/generate', json={'model':'qwen3:1.7b','prompt':'hi','stream':False}); print(r.json()['response'])"

:: Verify Whisper GPU
python -c "import whisper, torch; m=whisper.load_model('small.en'); print('whisper loaded on', 'CUDA' if torch.cuda.is_available() else 'CPU')"

:: Verify OWL loads on GPU
python -c "from transformers import Owlv2Processor, Owlv2ForObjectDetection; import torch; m=Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble').to('cuda'); print('OWL on GPU OK')"
```

All 5 should succeed with no error.

---

## Common Windows gotchas

| Symptom | Fix |
|---|---|
| `Microsoft Visual C++ 14.0 required` during pip install | Install "Build Tools for Visual Studio" from https://visualstudio.microsoft.com/downloads/ (C++ Build Tools) |
| Whisper says "no CUDA" but you have GPU | `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121` |
| `sounddevice` records silence | Windows mic permission: Settings → Privacy → Microphone → enable for desktop apps |
| Camera opens but returns black frames | Try different `CAMERA_ID` (0, 1, 2), or grant camera permission in Windows settings |
| `pymycobot` "COM3 access denied" | Close any other program using the port (Arduino IDE, etc.), or check Device Manager for correct COM number |
| Ollama not responding on port 11434 | Ollama service may not be running — search "Ollama" in Start menu and launch |

---

## Approximate download sizes / disk footprint

- PyTorch + CUDA: **~3 GB**
- Qwen3 via Ollama: **~1.4 GB**
- Whisper small.en: **~250 MB**
- OWLv2: **~600 MB** (downloads on first use)
- **Total: ~5.5 GB** for models + PyTorch

RAM usage at runtime: ~4–6 GB (Ollama + Whisper + OWL loaded together).
