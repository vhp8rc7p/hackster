# Windows Installation Guide (with NVIDIA GPU)

Getting the voice-controlled robot arm demo running on a Windows PC with
NVIDIA GPU. **Same models as the Mac version** (Qwen3-1.7B, Nemotron 3.5,
OWLv2) — just loaded through different Python libraries (MLX is Apple
Silicon only, so we use HuggingFace + NVIDIA NeMo on Windows/CUDA).

## Model → loader mapping

| Model | Mac (MLX) | Windows (this guide) |
|---|---|---|
| **Qwen3-1.7B** (LLM) | `mlx-lm` | HuggingFace `transformers` + CUDA |
| **Nemotron 3.5 streaming ASR** | `mlx-audio` | **NVIDIA NeMo** toolkit + CUDA |
| **OWLv2** (detection) | PyTorch + MPS | PyTorch + **CUDA** (same code) |
| Robot API | `pymycobot` | `pymycobot` (identical, port becomes `COMx`) |

---

## 1. Prerequisites

Install these first:

- **Python 3.11 (64-bit)** — https://python.org — check "Add Python to PATH"
- **Git for Windows** — https://git-scm.com
- **NVIDIA driver + CUDA Toolkit 12.1** — https://developer.nvidia.com/cuda-downloads
- **Microsoft Visual C++ Build Tools** — https://visualstudio.microsoft.com/downloads/ (needed by some pip packages)

Verify in cmd:
```cmd
python --version    :: 3.11.x
git --version
nvidia-smi          :: shows GPU + CUDA version
```

---

## 2. Clone repo + venv

```cmd
cd C:\Users\%USERNAME%\
git clone https://github.com/vhp8rc7p/hackster.git
cd hackster\local-ai-robot-arm
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## 3. Install PyTorch with CUDA (BEFORE anything else)

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```cmd
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
Must print `cuda: True <YourGPU>`.

---

## 4. Install Qwen3-1.7B via `transformers`

```cmd
pip install transformers accelerate sentencepiece
```

First run of Python will download the model (~3.4 GB) to `%USERPROFILE%\.cache\huggingface`. Pre-download it:

```cmd
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B'); AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B'); print('Qwen3 downloaded')"
```

**Loader replacement for `qwen_command.py`** — instead of:
```python
from mlx_lm import load as load_llm, generate as generate_llm
llm, tok = load_llm("Qwen/Qwen3-1.7B")
raw = generate_llm(llm, tok, prompt=prompt, max_tokens=120, verbose=False)
```
Use:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
llm = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", torch_dtype=torch.float16, device_map="cuda")

def generate_llm(llm, tok, prompt, max_tokens=120, verbose=False):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    outputs = llm.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

---

## 5. Install Nemotron 3.5 ASR via NVIDIA NeMo

```cmd
pip install nemo_toolkit[asr]
```

This is a bigger install (~2 GB of extra deps — NeMo pulls in a lot).
NeMo requires CUDA to run efficiently.

**Nemotron ASR model:** available on Hugging Face as
`nvidia/parakeet-tdt-0.6b-v2` (streaming ASR, English) or the specific
Nemotron variant if published. Pre-download:

```cmd
python -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.EncDecRNNTBPEModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2'); print('ASR model downloaded')"
```

**Loader replacement for `qwen_command.py`** — instead of:
```python
from mlx_audio.stt import load as load_stt
stt = load_stt("mlx-community/nemotron-3.5-asr-streaming-0.6b")
result = stt.generate(tmp_wav_path, language="en-US")
text = result.text
```
Use:
```python
import nemo.collections.asr as nemo_asr
stt = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    "nvidia/parakeet-tdt-0.6b-v2")
stt = stt.cuda().eval()

def transcribe(wav_path):
    hyps = stt.transcribe(audio=[wav_path], batch_size=1)
    return hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])
```

*Note: if you want the exact Nemotron-3.5 streaming variant, check
NVIDIA's NGC catalog (https://catalog.ngc.nvidia.com/) for the current
distribution — some Nemotron ASR models ship there rather than HF.*

---

## 6. Everything else

```cmd
pip install pillow numpy opencv-python
pip install sounddevice soundfile
pip install pymycobot
pip install ikpy scipy
```

OWLv2 uses `transformers` (already installed in step 4). Pre-download:

```cmd
python -c "from transformers import Owlv2Processor, Owlv2ForObjectDetection; Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble'); print('OWL downloaded')"
```

---

## 7. Robot serial port (Windows convention)

Windows uses `COMx` instead of `/dev/tty.usbserial-*`. Find yours:

1. Plug mycobot into USB
2. **Device Manager** → Ports (COM & LPT)
3. Look for "USB Serial Port (COMx)" — note the number

Test:
```cmd
python -c "from pymycobot.mycobot280 import MyCobot280; import time; mc=MyCobot280('COM3', 115200); time.sleep(2); print(mc.get_angles())"
```

In every script, update `SERIAL_PORT = "/dev/tty.usbserial-*"` → `SERIAL_PORT = "COM3"` (your actual port).

---

## 8. Camera (DirectShow backend)

Windows OpenCV needs the DirectShow backend explicitly:

```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # instead of cv2.VideoCapture(0)
```

Grant camera permission: **Settings → Privacy & security → Camera → enable "Let desktop apps access your camera"**.

Verify:
```cmd
python -c "import cv2; c=cv2.VideoCapture(0, cv2.CAP_DSHOW); ok,f=c.read(); print('OK' if ok else 'FAIL', f.shape if ok else '')"
```

---

## 9. Complete smoke test

After all above, run each check:

```cmd
:: Robot
python -c "from pymycobot.mycobot280 import MyCobot280; import time; mc=MyCobot280('COM3',115200); time.sleep(2); print('robot angles:', mc.get_angles())"

:: Camera
python -c "import cv2; c=cv2.VideoCapture(0, cv2.CAP_DSHOW); ok,f=c.read(); print('camera:', 'OK' if ok else 'FAIL')"

:: Qwen3 on GPU
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; import torch; m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype=torch.float16, device_map='cuda'); print('Qwen3 on', m.device)"

:: Nemotron ASR on GPU
python -c "import nemo.collections.asr as a; m=a.models.EncDecRNNTBPEModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2').cuda(); print('ASR on GPU')"

:: OWLv2 on GPU
python -c "from transformers import Owlv2ForObjectDetection; m=Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble').to('cuda'); print('OWL on GPU')"
```

All five must succeed.

---

## Common Windows gotchas

| Symptom | Fix |
|---|---|
| `Microsoft Visual C++ 14.0 required` | Install "Build Tools for Visual Studio" (C++ workload) |
| CUDA not detected after install | Restart PC, reinstall PyTorch with correct CUDA index URL |
| `nemo_toolkit` install fails | Requires C++ build tools + a lot of RAM — allow 15+ min |
| Mic records silence | Windows mic permission — Settings → Privacy → Microphone |
| Camera returns black frames | Try `CAMERA_ID = 1` or grant camera permission |
| `COM3 access denied` | Another program has the port (Arduino IDE etc.) |

---

## Disk footprint

- PyTorch + CUDA: ~3 GB
- Qwen3-1.7B: ~3.4 GB
- Nemotron/Parakeet ASR: ~0.7 GB
- NeMo toolkit deps: ~2 GB
- OWLv2: ~600 MB
- **Total: ~10 GB**

Runtime RAM: ~8–10 GB (all three models loaded).

---

## If NeMo is too heavy (optional fallback)

`nemo_toolkit` is a big install. If it fails or the client's PC doesn't
have space, you can substitute Nemotron ASR with **OpenAI Whisper**
(smaller, easier install, cross-platform, GPU-accelerated):

```cmd
pip install openai-whisper
```
```python
import whisper
stt = whisper.load_model("small.en")   # ~250 MB
result = stt.transcribe(wav_path, language="en", fp16=True)
text = result["text"]
```
Whisper `small.en` is ~250 MB vs Nemotron's ~700 MB and roughly comparable
quality for short English commands.

Only use as fallback — the primary path (Sections 5) keeps the same
model as the Mac version.
