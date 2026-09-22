#!/usr/bin/env python3
"""
Microphone + speech-recognition test bench.

Mirrors the exact capture + STT pipeline used by qwen_command.py so you can
measure how well a NEW mic works before wiring it into the real demo:

  sounddevice capture @ 16kHz mono
    -> RMS VAD gate (start on loud, end on trailing silence)
      -> WAV -> Nemotron ASR (mlx-audio) .generate(wav, language="en-US")
        -> compare transcript to the expected phrase

It reports, per phrase and overall:
  - whether the transcript matched (exact + word-error-rate)
  - the speech RMS level (so you can see if VAD_RMS_THRESHOLD fits this mic)
  - the ambient noise floor (measured up front)
and prints the device index/name to hardcode into qwen_command.py.

USAGE
  # list input devices, then quit:
  ./mlx_env/bin/python test_mic.py --list

  # run the full test on a chosen device (index from --list, or a name substring):
  ./mlx_env/bin/python test_mic.py --device 2
  ./mlx_env/bin/python test_mic.py --device "Yeti"

  # run on the system default input:
  ./mlx_env/bin/python test_mic.py

  # tweak which model / phrase set:
  ./mlx_env/bin/python test_mic.py --model parakeet
  ./mlx_env/bin/python test_mic.py --phrases quick
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

# --- capture params: MUST match qwen_command.py so the test is representative ---
SAMPLE_RATE = 16000
VAD_FRAME_S = 0.05
VAD_RMS_THRESHOLD = 0.04     # same default as the real pipeline
VAD_SILENCE_END_S = 0.9       # trailing silence to end (was 0.7 — too eager,
                              # cut off natural mid-phrase pauses)
VAD_MIN_UTTERANCE_S = 0.25
VAD_MAX_UTTERANCE_S = 8.0
# --- robustness knobs (new; not in the original pipeline) ---
VAD_WARMUP_S = 0.25           # discard stream-open transient
VAD_PREROLL_S = 0.30          # audio kept before trigger, so onsets aren't clipped
VAD_MIN_VOICED_S = 0.20       # min loud audio required, else it's a false trigger

STT_OPTIONS = {
    "nemotron": "mlx-community/nemotron-3.5-asr-streaming-0.6b",
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v2",
}

# Phrases the real command parser expects to hear. These are what actually
# matters — a mic that nails "the weather today" but flubs "pick the green
# cube" is useless here, so we test the real vocabulary.
PHRASE_SETS = {
    "quick": [
        "pick the green cube",
        "home",
        "stop",
    ],
    "commands": [
        "pick the green cube",
        "pick the blue cube",
        "place it in the cardboard box",
        "bring the pink cube to my hand",
        "put the yellow cube on my hand",
        "home",
        "stop",
        "wait",
    ],
}

# Expected transcript for the pre-generated command WAVs (filenames have no
# spaces). Keyed by the lowercase basename without extension. Files not listed
# here are transcribed but shown unscored (or add a same-named .txt sidecar).
WAV_EXPECTED = {
    "pickthegreencube": "pick the green cube",
    "pickthebluecube": "pick the blue cube",
    "handmethegreencube": "hand me the green cube",
    "handmethebluecube": "hand me the blue cube",
    "handmeoverthecreditcard": "hand me over the credit card",
    "dropitinthebox": "drop it in the box",
    "canyougiveittome": "can you give it to me",
    "waitcanyougiveittome": "wait can you give it to me",
    "wait": "wait",
    "placethegreencubeinthecardboardbox": "place the green cube in the cardboard box",
    "putgreencubeinthecardboardbox": "put green cube in the cardboard box",
}


def _expected_for(path):
    """Expected transcript for a wav: a same-named .txt sidecar wins, else the
    WAV_EXPECTED table, else None (unscored)."""
    base = os.path.splitext(os.path.basename(path))[0]
    sidecar = os.path.splitext(path)[0] + ".txt"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            return f.read().strip()
    return WAV_EXPECTED.get(base.lower())


# --------------------------------------------------------------------------- #
# text scoring
# --------------------------------------------------------------------------- #
def _norm(s):
    """Lowercase, drop punctuation, collapse whitespace."""
    s = s.lower()
    keep = "abcdefghijklmnopqrstuvwxyz0123456789 "
    s = "".join(c if c in keep else " " for c in s)
    return " ".join(s.split())


def _wer(ref, hyp):
    """Word error rate via Levenshtein distance over word lists (0.0 = perfect)."""
    r = _norm(ref).split()
    h = _norm(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # DP edit distance
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,        # deletion
                          d[i, j - 1] + 1,        # insertion
                          d[i - 1, j - 1] + cost)  # substitution
    return d[len(r), len(h)] / len(r)


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def list_devices():
    print("\nInput devices (max_input_channels > 0):\n")
    default_in = None
    try:
        default_in = sd.default.device[0]
    except Exception:
        pass
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        star = "  <- default" if i == default_in else ""
        print(f"  [{i:2}] {d['name']}  "
              f"(in ch={d['max_input_channels']}, "
              f"native sr={int(d.get('default_samplerate', 0))}){star}")
    print()


def resolve_device(spec):
    """spec: None (system default), an int index, or a name substring."""
    if spec is None:
        return None, "system default"
    # integer index?
    try:
        idx = int(spec)
        d = sd.query_devices(idx)
        if d.get("max_input_channels", 0) <= 0:
            sys.exit(f"Device [{idx}] '{d['name']}' has no input channels.")
        return idx, d["name"]
    except ValueError:
        pass
    # name substring, case-insensitive
    matches = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
               if d.get("max_input_channels", 0) > 0
               and spec.lower() in d["name"].lower()]
    if not matches:
        sys.exit(f"No input device name contains '{spec}'. Try --list.")
    if len(matches) > 1:
        print(f"Multiple matches for '{spec}':")
        for i, n in matches:
            print(f"  [{i}] {n}")
        sys.exit("Be more specific, or pass the index.")
    return matches[0]


def measure_noise_floor(device_idx, seconds=2.0):
    """Record ambient silence and return its RMS."""
    n = int(SAMPLE_RATE * seconds)
    rec = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                 device=device_idx)
    sd.wait()
    return float(np.sqrt(np.mean(rec.flatten() ** 2)))


def capture_utterance(device_idx, timeout_s=10.0):
    """Robust VAD capture. Returns (audio_float32, speech_rms, peak_rms) or
    (None, 0, 0) on timeout.

    Improvements over the naive gate (which clipped word onsets and returned
    empty clips on stray clicks):
      - WARMUP: discard the first few chunks so the stream-open transient
        (pop/click when the device starts) can't false-trigger.
      - PRE-ROLL: keep a rolling buffer of audio from *before* the trigger and
        prepend it, so the first phoneme isn't cut off.
      - MIN VOICED: only finalize an utterance once we've seen enough genuinely
        loud (voiced) chunks — a lone click surrounded by silence is discarded
        and listening continues, instead of being returned as "".
    """
    chunk = int(SAMPLE_RATE * VAD_FRAME_S)
    silence_end = int(VAD_SILENCE_END_S / VAD_FRAME_S)
    max_chunks = int(VAD_MAX_UTTERANCE_S / VAD_FRAME_S)
    warmup_chunks = int(VAD_WARMUP_S / VAD_FRAME_S)
    preroll_chunks = int(VAD_PREROLL_S / VAD_FRAME_S)
    min_voiced = int(VAD_MIN_VOICED_S / VAD_FRAME_S)

    from collections import deque
    preroll = deque(maxlen=preroll_chunks)
    buffer, silence, speaking = [], 0, False
    voiced = 0                       # count of loud chunks in this utterance
    speech_rms_vals, peak = [], 0.0
    start = time.time()

    def reset():
        nonlocal buffer, silence, speaking, voiced, speech_rms_vals
        buffer, silence, speaking, voiced, speech_rms_vals = [], 0, False, 0, []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=chunk, device=device_idx) as stream:
        # Drain the stream-open transient before we start listening for speech.
        for _ in range(warmup_chunks):
            stream.read(chunk)

        while True:
            if not speaking and (time.time() - start) > timeout_s:
                # Report the loudest level seen so a missed clip shows how close
                # it came to the gate (helps position/aim the phone).
                return None, 0.0, peak
            data, _ = stream.read(chunk)
            c = data.flatten()
            rms = float(np.sqrt(np.mean(c ** 2)))
            peak = max(peak, rms)

            if rms > VAD_RMS_THRESHOLD:
                if not speaking:
                    speaking = True
                    buffer.extend(preroll)   # prepend pre-roll for onset
                silence = 0
                voiced += 1
                buffer.append(c)
                speech_rms_vals.append(rms)
            elif speaking:
                buffer.append(c)             # keep trailing audio (hangover)
                silence += 1
                if silence >= silence_end or len(buffer) >= max_chunks:
                    # Require enough actual speech, else it was a false trigger.
                    if voiced < min_voiced:
                        reset()
                        continue
                    audio = np.concatenate(buffer)
                    srms = float(np.mean(speech_rms_vals)) if speech_rms_vals else 0.0
                    return audio, srms, peak
            else:
                preroll.append(c)            # not speaking yet — remember recent audio


def transcribe(stt, audio):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    tmp.close()
    try:
        result = stt.generate(tmp.name, language="en-US")
        return (getattr(result, "text", "") or "").strip()
    finally:
        os.unlink(tmp.name)


def _level_flag(srms, peak, noise):
    """One-word verdict on the captured level, for phone/mic positioning."""
    if peak >= 0.95:
        return "⚠ CLIPPING — lower volume"
    if noise > 0 and srms > 0 and srms < noise * 2:
        return "⚠ LOW — raise volume / aim closer"
    if peak > 0.6:
        return "hot"
    return "ok"


# --------------------------------------------------------------------------- #
# direct-from-wav test (no mic, no room — tests the STT model's ceiling)
# --------------------------------------------------------------------------- #
def run_from_wav(wav_dir, model_key):
    import glob
    paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not paths:
        sys.exit(f"No .wav files in {wav_dir!r}")

    print("=" * 64)
    print("  DIRECT-FROM-WAV TEST  (bypasses mic + room — STT model ceiling)")
    print("=" * 64)
    print(f"  Dir   : {wav_dir}")
    print(f"  Model : {model_key}  ({STT_OPTIONS[model_key]})")
    print(f"  Files : {len(paths)}")
    print("=" * 64)

    print(f"\nLoading STT ({model_key})...")
    from mlx_audio.stt import load as load_stt
    stt = load_stt(STT_OPTIONS[model_key])
    print("  ✓ model loaded\n")

    scored, results = [], []
    for p in paths:
        # mlx-audio reads and resamples the file itself; pass the path directly.
        try:
            result = stt.generate(p, language="en-US")
            text = (getattr(result, "text", "") or "").strip()
        except Exception as e:
            text = f"<error: {e}>"
        exp = _expected_for(p)
        name = os.path.basename(p)
        if exp is None:
            print(f"  ?  {name[:34]:34} -> {text}")
            results.append((name, text, None, None))
            continue
        wer = _wer(exp, text)
        exact = _norm(exp) == _norm(text)
        mark = "✓" if exact else ("~" if wer <= 0.34 else "✗")
        print(f"  {mark}  {name[:34]:34} -> {text}   ({wer:.0%})")
        results.append((name, text, wer, exact))
        scored.append((wer, exact))

    if scored:
        n = len(scored)
        exact_n = sum(1 for _, e in scored if e)
        near_n = sum(1 for w, e in scored if (not e) and w <= 0.34)
        avg = float(np.mean([w for w, _ in scored]))
        print("\n" + "=" * 64)
        print(f"  Exact  : {exact_n}/{n} ({exact_n/n:.0%})   "
              f"Usable: {exact_n + near_n}/{n} ({(exact_n + near_n)/n:.0%})   "
              f"Avg WER: {avg:.0%}")
        print("  (This is the model's BEST case on clean audio. Live mic capture")
        print("   will be equal or worse — it can't beat this ceiling.)")
        print("=" * 64)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global VAD_RMS_THRESHOLD
    ap = argparse.ArgumentParser(description="Mic + STT test bench")
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    ap.add_argument("--device", default=None,
                    help="input device: index (from --list) or name substring")
    ap.add_argument("--model", default="nemotron", choices=list(STT_OPTIONS),
                    help="STT model to test (default: nemotron)")
    ap.add_argument("--phrases", default="commands", choices=list(PHRASE_SETS),
                    help="phrase set: 'commands' (8) or 'quick' (3)")
    ap.add_argument("--save-wavs", metavar="DIR", default=None,
                    help="also save each recording to DIR for later inspection")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"override VAD RMS trigger (default {VAD_RMS_THRESHOLD}); "
                         "lower it for a quiet mic")
    ap.add_argument("--from-wav", metavar="DIR", default=None,
                    help="skip the mic; transcribe every .wav in DIR directly "
                         "(tests the STT model's ceiling on clean audio)")
    ap.add_argument("--play-wavs", metavar="DIR", default=None,
                    help="live mic test, but prompts you to PLAY each .wav in DIR "
                         "(e.g. from your phone) instead of speaking; scores "
                         "against each file's known phrase")
    args = ap.parse_args()

    if args.threshold is not None:
        VAD_RMS_THRESHOLD = args.threshold

    if args.list:
        list_devices()
        return

    if args.from_wav:
        run_from_wav(args.from_wav, args.model)
        return

    device_idx, device_name = resolve_device(args.device)
    if args.play_wavs:
        import glob
        paths = sorted(glob.glob(os.path.join(args.play_wavs, "*.wav")))
        if not paths:
            sys.exit(f"No .wav files in {args.play_wavs!r}")
        phrases = [_expected_for(p) or os.path.splitext(os.path.basename(p))[0]
                   for p in paths]
        prompts = [f'PLAY on your phone:  {os.path.basename(p)}' for p in paths]
        set_label = f"play-wavs: {args.play_wavs}"
    else:
        phrases = PHRASE_SETS[args.phrases]
        prompts = [f'say:  "{ph}"' for ph in phrases]
        set_label = f"{args.phrases} set"

    print("=" * 64)
    print("  MIC + SPEECH RECOGNITION TEST")
    print("=" * 64)
    print(f"  Device : [{device_idx}] {device_name}"
          if device_idx is not None else f"  Device : {device_name}")
    print(f"  Model  : {args.model}  ({STT_OPTIONS[args.model]})")
    print(f"  Phrases: {len(phrases)} ({set_label})")
    print(f"  VAD RMS threshold: {VAD_RMS_THRESHOLD}")
    print("=" * 64)

    if args.save_wavs:
        os.makedirs(args.save_wavs, exist_ok=True)

    # 1) noise floor
    print("\n[1/3] Measuring ambient noise floor — stay QUIET for 2s...")
    noise = measure_noise_floor(device_idx)
    margin = VAD_RMS_THRESHOLD / noise if noise > 0 else float("inf")
    print(f"      noise floor RMS = {noise:.4f}   "
          f"(threshold is {margin:.1f}x above the noise)")
    if noise >= VAD_RMS_THRESHOLD:
        print("      ⚠ Noise floor is ABOVE the VAD threshold — the mic will")
        print("        trigger on silence. Lower the gain or move the mic.")
    elif margin < 2.0:
        print("      ⚠ Little headroom over noise — quiet speech may be missed.")
    else:
        print("      ✓ Healthy gap between noise and the trigger threshold.")

    # 2) load model
    print(f"\n[2/3] Loading STT ({args.model}) — first run downloads a few GB...")
    from mlx_audio.stt import load as load_stt
    stt = load_stt(STT_OPTIONS[args.model])
    print("      ✓ model loaded")

    # 3) phrase loop
    if args.play_wavs:
        print(f"\n[3/3] For each item: press ENTER, then PLAY that file on your")
        print(f"      phone near the mic. Recording auto-stops on silence.\n")
    else:
        print(f"\n[3/3] Speak each phrase after the prompt. Press ENTER when ready,")
        print(f"      then say it once, clearly. Recording auto-stops on silence.\n")

    results = []  # (phrase, transcript, wer, exact, speech_rms, peak)
    for n, (phrase, prompt) in enumerate(zip(phrases, prompts), 1):
        input(f'  ({n}/{len(phrases)}) Press ENTER, then {prompt}')
        print("      ● listening...", end="", flush=True)
        audio, srms, peak = capture_utterance(device_idx)
        if audio is None:
            # peak = loudest level seen; show how close it came to the gate.
            gap = f"peak {peak:.3f} vs gate {VAD_RMS_THRESHOLD:.3f}"
            hint = ("→ never crossed the gate; raise volume / aim phone at mic"
                    if peak < VAD_RMS_THRESHOLD else
                    "→ crossed the gate but too brief; play the whole clip")
            print(f" TIMEOUT ({gap})  {hint}")
            results.append((phrase, "<no speech>", 1.0, False, 0.0, peak))
            continue
        if args.save_wavs:
            sf.write(os.path.join(args.save_wavs, f"phrase_{n:02}.wav"),
                     audio, SAMPLE_RATE)
        text = transcribe(stt, audio)
        wer = _wer(phrase, text)
        exact = _norm(phrase) == _norm(text)
        results.append((phrase, text, wer, exact, srms, peak))
        mark = "✓" if exact else ("~" if wer <= 0.34 else "✗")
        level = _level_flag(srms, peak, noise)
        print(f"\r      {mark} heard: \"{text}\"   "
              f"(WER {wer:.0%}, level mean {srms:.3f} peak {peak:.3f} — {level})")

    # summary
    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    exact_n = sum(1 for r in results if r[3])
    near_n = sum(1 for r in results if (not r[3]) and r[2] <= 0.34)
    avg_wer = float(np.mean([r[2] for r in results])) if results else 1.0
    speech_rmss = [r[4] for r in results if r[4] > 0]
    avg_speech = float(np.mean(speech_rmss)) if speech_rmss else 0.0
    peaks = [r[5] for r in results if r[5] > 0]
    avg_peak = float(np.mean(peaks)) if peaks else 0.0
    min_peak = min(peaks) if peaks else 0.0

    print(f"\n  {'PHRASE':30} {'HEARD':28} {'mean':>6} {'peak':>6}")
    for phrase, text, wer, exact, srms, peak in results:
        mark = "✓" if exact else ("~" if wer <= 0.34 else "✗")
        print(f"  {mark} {phrase[:28]:28} -> {text[:26]:26} "
              f"{srms:6.3f} {peak:6.3f} ({wer:.0%})")

    n = len(results)
    print(f"\n  Exact matches : {exact_n}/{n}  ({exact_n/n:.0%})")
    print(f"  Near matches  : {near_n}/{n}  (WER <= 34%, usually still parses)")
    print(f"  Usable total  : {exact_n + near_n}/{n}  ({(exact_n + near_n)/n:.0%})")
    print(f"  Average WER   : {avg_wer:.0%}")
    print(f"  Noise floor   : {noise:.4f}   (VAD gate {VAD_RMS_THRESHOLD})")
    if noise > 0:
        print(f"  Speech level  : mean {avg_speech:.4f} ({avg_speech/noise:.0f}x noise), "
              f"peak avg {avg_peak:.4f}, quietest peak {min_peak:.4f}")
        # If the quietest captured clip barely cleared the gate, playback is marginal.
        if min_peak > 0 and min_peak < VAD_RMS_THRESHOLD * 1.5:
            print(f"  ⚠ Quietest clip peaked at {min_peak:.3f}, close to the "
                  f"{VAD_RMS_THRESHOLD} gate — raise phone volume, aim at mic,")
            print(f"    or lower the gate (e.g. --threshold {max(round(min_peak*0.6,3),0.02)}).")

    # actionable advice
    print("\n  ---")
    if device_idx is not None:
        auto = ("MacBook" in device_name)
        if not auto:
            print(f"  NOTE: qwen_command.py auto-pins to a mic whose name contains")
            print(f"        \"MacBook\". This device won't be auto-selected. To use it,")
            print(f"        set the listener to device index {device_idx} "
                  f"(\"{device_name}\").")
    if avg_speech > 0 and noise > 0:
        suggested = round(min(avg_speech, max(noise * 3, VAD_RMS_THRESHOLD)) , 3)
        lo = round(noise * 3, 3)
        hi = round(avg_speech * 0.5, 3)
        if lo < hi:
            print(f"  TUNE: a VAD_RMS_THRESHOLD around {lo}-{hi} would sit safely")
            print(f"        between this mic's noise ({noise:.3f}) and speech "
                  f"({avg_speech:.3f}).")
    print("=" * 64)


if __name__ == "__main__":
    main()
