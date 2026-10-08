#!/usr/bin/env python3
"""
Analyse the recorded voice clips to find WHY transcription is failing.

qwen_command saves every captured utterance to voice_clips/ and logs what it
transcribed to in voice.csv. Text alone says the transcript was wrong; the
audio says why. For each clip this reports:

  SNR          speech vs the quiet parts of the same clip
  onset        how much speech sits in the first 150ms (clipped start?)
  tail         did it get cut off mid-word
  clipping     samples at full scale (gain too hot)
  pump tone    energy at the pump's harmonics (110/328/657/876 Hz)
  re-transcribe  what Nemotron makes of it now, incl. with denoising

Then it groups the failures so the dominant cause is obvious.

USAGE
  ./mlx_env/bin/python analyze_voice.py              # all clips
  ./mlx_env/bin/python analyze_voice.py --last 10
  ./mlx_env/bin/python analyze_voice.py --no-stt     # skip re-transcription
"""

import argparse
import csv
import glob
import os
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
AUDIO_DIR = "voice_clips"
VOICE_LOG = "voice.csv"
PUMP_TONES = [110, 328, 657, 767, 876]      # measured pump harmonics


def load_log():
    """clip filename -> what it transcribed to (or '' if empty)."""
    out = {}
    if not os.path.exists(VOICE_LOG):
        return out
    for r in csv.DictReader(open(VOICE_LOG)):
        t = r.get("text", "")
        if "voice_clips" not in t:
            continue
        clip, _, said = t.partition(" | ")
        out[os.path.basename(clip.strip())] = said.strip()
    return out


def analyse(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    n = len(x)
    fr = int(SR * 0.02)
    rms = np.array([np.sqrt(np.mean(x[i:i + fr] ** 2))
                    for i in range(0, max(n - fr, 1), fr)])
    if len(rms) < 3:
        return None
    loud = np.percentile(rms, 90)
    quiet = np.percentile(rms, 20)
    snr = 20 * np.log10(loud / max(quiet, 1e-6))

    onset_frames = max(1, int(0.15 / 0.02))
    onset = float(np.mean(rms[:onset_frames]) / max(loud, 1e-9))
    tail = float(np.mean(rms[-onset_frames:]) / max(loud, 1e-9))
    clip_pct = float(np.mean(np.abs(x) > 0.98) * 100)

    # how much energy sits on the pump harmonics vs nearby speech bands
    f = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    pump = sum(float(f[(freqs > t - 12) & (freqs < t + 12)].mean())
               for t in PUMP_TONES) / len(PUMP_TONES)
    speech = float(f[(freqs > 300) & (freqs < 3400)].mean())
    pump_ratio = pump / max(speech, 1e-9)

    return {"dur": n / SR, "snr": snr, "loud": loud, "quiet": quiet,
            "onset": onset, "tail": tail, "clip": clip_pct,
            "pump_ratio": pump_ratio}


def verdict(a):
    """Most likely reason this clip would transcribe badly."""
    if a["clip"] > 0.5:
        return "CLIPPING (gain too hot)"
    if a["snr"] < 10:
        return "LOW SNR (noise ~ speech)"
    if a["onset"] > 0.55:
        return "CLIPPED ONSET (speech at frame 0)"
    if a["tail"] > 0.55:
        return "CUT OFF (still talking at the end)"
    if a["pump_ratio"] > 1.5:
        return "PUMP TONES dominate"
    if a["loud"] < 0.03:
        return "TOO QUIET"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--no-stt", action="store_true")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    if not clips:
        print(f"No clips in {AUDIO_DIR}/ — run qwen_command.py and speak first.")
        return
    if args.last:
        clips = clips[-args.last:]
    log = load_log()

    stt = None
    if not args.no_stt:
        print("loading STT for re-transcription...")
        from mlx_audio.stt import load as load_stt
        stt = load_stt("mlx-community/nemotron-3.5-asr-streaming-0.6b")

    def tx(a):
        t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(t.name, a, SR)
        t.close()
        try:
            r = stt.generate(t.name, language="en-US")
            return (getattr(r, "text", "") or "").strip()
        finally:
            os.unlink(t.name)

    print("=" * 78)
    print(f"  VOICE CLIP ANALYSIS — {len(clips)} clips")
    print("=" * 78)
    print(f"  {'clip':16} {'dur':>5} {'SNR':>6} {'lvl':>6} {'onset':>6} "
          f"{'tail':>5} {'pump':>5}  verdict")
    print("  " + "-" * 74)

    causes = {}
    rows = []
    for p in clips:
        a = analyse(p)
        if a is None:
            continue
        v = verdict(a)
        causes[v] = causes.get(v, 0) + 1
        rows.append((p, a, v))
        print(f"  {os.path.basename(p):16} {a['dur']:5.1f} {a['snr']:5.1f}dB "
              f"{a['loud']:6.3f} {a['onset']:6.2f} {a['tail']:5.2f} "
              f"{a['pump_ratio']:5.2f}  {v}")
        said = log.get(os.path.basename(p))
        if said is not None:
            print(f"      logged -> {said!r}" if said else "      logged -> (empty)")
        if stt is not None:
            x, _ = sf.read(p)
            if x.ndim > 1:
                x = x.mean(axis=1)
            again = tx(x.astype(np.float32))
            print(f"      now    -> {again!r}")

    print("\n  DOMINANT CAUSES")
    for c, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"    {n:3d}  {c}")

    if rows:
        snrs = [r[1]["snr"] for r in rows]
        lv = [r[1]["loud"] for r in rows]
        print(f"\n  median SNR {sorted(snrs)[len(snrs)//2]:.1f}dB   "
              f"median speech level {sorted(lv)[len(lv)//2]:.3f}")
        print("  (SNR under ~15dB is where ASR starts degrading badly)")
    print("=" * 78)


if __name__ == "__main__":
    main()
