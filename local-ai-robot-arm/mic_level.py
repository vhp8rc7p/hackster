#!/usr/bin/env python3
"""
Live mic level meter — for setting the input gain correctly.

Open System Settings -> Sound -> Input, then watch this while you drag the
input volume slider.

TARGETS
  silence  : bar should sit well under the [gate] marker  (~0.015)
  speaking : bar should push past it clearly              (~0.08-0.12)

If silence is already at or above the gate, the VAD can never detect the end
of an utterance and speech recognition stops working entirely.

  ./mlx_env/bin/python mic_level.py
  ./mlx_env/bin/python mic_level.py --device 1
"""
import argparse

import numpy as np
import sounddevice as sd

SR = 16000
FRAME = 0.05
GATE = 0.04          # VAD_RMS_THRESHOLD in qwen_command.py
PUMP_GATE = 0.075    # VAD_PUMP_THRESHOLD (used while the pump runs)
WIDTH = 50
FULL = 0.20          # bar full-scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()

    ins = [(i, d) for i, d in enumerate(sd.query_devices())
           if d.get("max_input_channels", 0) > 0]
    if not ins:
        raise SystemExit("no input devices")
    idx = args.device if args.device is not None else ins[0][0]
    name = sd.query_devices()[idx]["name"]

    g = int(GATE / FULL * WIDTH)
    pg = int(PUMP_GATE / FULL * WIDTH)
    print(f"mic [{idx}] {name}")
    print(f"gate={GATE}  pump-gate={PUMP_GATE}   Ctrl-C to stop\n")
    scale = [" "] * WIDTH
    scale[g] = "|"; scale[min(pg, WIDTH - 1)] = "|"
    print("     " + "".join(scale) + "   <- gate / pump-gate")

    recent = []
    n = int(SR * FRAME)
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=n, device=idx) as s:
            while True:
                data, _ = s.read(n)
                rms = float(np.sqrt(np.mean(data.flatten() ** 2)))
                recent.append(rms)
                if len(recent) > 40:
                    recent.pop(0)
                filled = max(0, min(WIDTH, int(rms / FULL * WIDTH)))
                bar = "".join("#" if i < filled else
                              ("|" if i in (g, pg) else "-")
                              for i in range(WIDTH))
                if rms >= PUMP_GATE:
                    tag = "LOUD "
                elif rms >= GATE:
                    tag = "over "
                else:
                    tag = "quiet"
                print(f"\r{rms:6.4f} [{bar}] {tag} avg{np.mean(recent):6.4f}",
                      end="", flush=True)
    except KeyboardInterrupt:
        print(f"\n\nmean over last samples: {np.mean(recent):.4f}")
        if np.mean(recent) >= GATE:
            print("⚠ Idle level is AT/ABOVE the gate — lower the input volume in")
            print("  System Settings > Sound > Input until this reads ~0.015.")
        else:
            print("✓ Level looks usable.")


if __name__ == "__main__":
    main()
