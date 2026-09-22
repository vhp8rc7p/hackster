#!/usr/bin/env python3
"""
Measure mic noise floor with the suction pump OFF vs ON, for every input device.

Why this matters: the pump motor hum can raise a mic's noise floor above the
VAD trigger (VAD_RMS_THRESHOLD in qwen_command.py). If it does, the listener
will false-trigger on the pump — or the pump will mask your voice — *while the
arm is picking*. This tool tells you, per mic:

  - the noise floor with the pump OFF (baseline) and ON
  - how much the pump adds (delta)
  - whether the pump-on floor crosses the VAD gate  → false-trigger risk

It measures ALL input devices each pass, so external vs built-in are compared
under identical pump conditions.

USAGE
  # auto-drive the pump over serial (finds the port), measure every mic:
  ./mlx_env/bin/python test_noise.py

  # you toggle the pump by hand (script just prompts and measures):
  ./mlx_env/bin/python test_noise.py --manual

  # pin the robot serial port, longer capture, only certain mics:
  ./mlx_env/bin/python test_noise.py --port /dev/tty.usbserial-XXXX --seconds 4
  ./mlx_env/bin/python test_noise.py --devices 0,2
"""

import argparse
import glob
import sys
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
VAD_RMS_THRESHOLD = 0.04     # keep in sync with qwen_command.py
FRAME_S = 0.05
PUMP_PIN = 2                 # from qwen_command.py / pump_off.py
VALVE_PIN = 5


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def input_devices():
    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d.get("max_input_channels", 0) > 0]


def measure(device_idx, seconds):
    """Return (mean_rms, peak_frame_rms) over `seconds` of capture."""
    n = int(SAMPLE_RATE * seconds)
    rec = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                 device=device_idx)
    sd.wait()
    x = rec.flatten()
    mean_rms = float(np.sqrt(np.mean(x ** 2)))
    fr = int(SAMPLE_RATE * FRAME_S)
    frames = [x[i:i + fr] for i in range(0, max(len(x) - fr, 1), fr)]
    peak = max((float(np.sqrt(np.mean(f ** 2))) for f in frames), default=mean_rms)
    return mean_rms, peak


# --------------------------------------------------------------------------- #
# pump
# --------------------------------------------------------------------------- #
def find_port():
    ports = sorted(glob.glob("/dev/tty.usbserial-*") +
                   glob.glob("/dev/tty.usbmodem*"))
    return ports[0] if ports else None


def pump_connect(port):
    from pymycobot.mycobot280 import MyCobot280
    mc = MyCobot280(port, 115200)
    time.sleep(1.0)
    return mc


def pump_set(mc, on):
    # pins: 0/0 = ON (suction), 1/1 = OFF — matches pump_off.py
    if on:
        mc.set_basic_output(VALVE_PIN, 0)
        mc.set_basic_output(PUMP_PIN, 0)
    else:
        mc.set_basic_output(PUMP_PIN, 1)
        mc.set_basic_output(VALVE_PIN, 1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Pump noise-floor test")
    ap.add_argument("--manual", action="store_true",
                    help="don't drive the pump over serial; prompt you to toggle it")
    ap.add_argument("--port", default=None,
                    help="robot serial port (default: auto-detect)")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="capture length per measurement (default 3s)")
    ap.add_argument("--devices", default=None,
                    help="comma-separated input device indices (default: all)")
    args = ap.parse_args()

    devs = input_devices()
    if args.devices:
        want = {int(x) for x in args.devices.split(",")}
        devs = [(i, d) for i, d in devs if i in want]
    if not devs:
        sys.exit("No input devices found (check --devices).")

    print("=" * 70)
    print("  PUMP NOISE-FLOOR TEST")
    print("=" * 70)
    print("  Mics under test:")
    for i, d in devs:
        print(f"    [{i}] {d['name']}")
    print(f"  Capture: {args.seconds}s per measurement   VAD gate: {VAD_RMS_THRESHOLD}")
    print("=" * 70)

    # connect pump (unless manual)
    mc = None
    if not args.manual:
        port = args.port or find_port()
        if not port:
            print("\n⚠ No serial port found. Falling back to --manual mode.")
            args.manual = True
        else:
            print(f"\nConnecting to robot on {port} ...")
            try:
                mc = pump_connect(port)
                print("  ✓ connected")
            except Exception as e:
                print(f"  ⚠ could not connect ({e}); falling back to manual mode.")
                args.manual = True

    def set_pump(on):
        state = "ON" if on else "OFF"
        if args.manual:
            input(f"\n  >>> Turn the pump {state}, then press ENTER to measure...")
        else:
            pump_set(mc, on)
            print(f"\n  pump {state}; letting it settle...")
            time.sleep(1.5)

    def measure_all(label):
        print(f"\n[{label}] measuring {len(devs)} mic(s) — keep the room quiet, "
              "don't speak...")
        out = {}
        for i, d in devs:
            mean_rms, peak = measure(i, args.seconds)
            out[i] = (mean_rms, peak)
            print(f"    [{i}] {d['name'][:28]:28}  "
                  f"mean {mean_rms:.4f}   peak {peak:.4f}")
        return out

    try:
        # baseline
        set_pump(False)
        off = measure_all("PUMP OFF")
        # pump on
        set_pump(True)
        on = measure_all("PUMP ON")
    finally:
        if mc is not None:
            try:
                pump_set(mc, False)
                print("\n  pump returned to OFF.")
            except Exception:
                pass
        elif not args.manual:
            pass
        else:
            print("\n  >>> You can turn the pump OFF now.")

    # report
    print("\n" + "=" * 70)
    print("  RESULTS  (mean RMS)")
    print("=" * 70)
    name_by = {i: d["name"] for i, d in devs}
    print(f"  {'mic':30} {'off':>8} {'on':>8} {'delta':>8} {'on vs gate':>12}")
    print("  " + "-" * 66)
    for i, _ in devs:
        o = off[i][0]
        n = on[i][0]
        delta = n - o
        ratio = n / VAD_RMS_THRESHOLD
        flag = "OVER GATE" if n >= VAD_RMS_THRESHOLD else "ok"
        print(f"  [{i}] {name_by[i][:25]:25} {o:8.4f} {n:8.4f} "
              f"{delta:+8.4f} {ratio:6.1f}x {flag:>4}")

    print("\n  Interpretation:")
    print(f"    - 'on' is the noise floor while the pump runs.")
    print(f"    - If 'on' is OVER GATE ({VAD_RMS_THRESHOLD}), that mic will")
    print(f"      false-trigger on the pump or mask your voice during a pick.")
    print(f"    - Lower delta = better pump isolation. Prefer the mic with the")
    print(f"      lowest 'on' floor and the most headroom under the gate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
