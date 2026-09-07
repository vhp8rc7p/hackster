"""Pose the arm by hand and capture the joint angles.

    python teach.py                 free-form: press Enter to print current angles
    python teach.py --pickplace     guided: capture the four pick-and-place poses

Releases the servos so the arm is limp. Hold the arm before it goes limp -- it
will sag under its own weight.
"""

import argparse
import time

import config
from pymycobot import MyCobot280

PICKPLACE_STEPS = [
    ("start_angles",
     "APPROACH pose -- above the cube, gripper lined up to come straight down"),
    ("pick",
     "AT the cube -- jaws around it, at the height where closing would grip it"),
    ("place",
     "OVER THE BOX at release height -- where you want the cube let go"),
]


def connect() -> MyCobot280:
    mc = MyCobot280(config.PORT, config.BAUD)
    time.sleep(0.6)
    mc.power_on()
    return mc


def read_angles(mc: MyCobot280, attempts: int = 5):
    for _ in range(attempts):
        a = mc.get_angles()
        if isinstance(a, list) and len(a) >= 6:
            return a
        time.sleep(0.3)
    return None


def free_form(mc: MyCobot280) -> None:
    print("Servos released. Pose the arm, press Enter to read. Ctrl-C to quit.\n")
    while True:
        input()
        angles = read_angles(mc)
        if angles is None:
            print("  (dropped frame, try again)")
            continue
        print(f'  "my_pose": [{", ".join(f"{a:.1f}" for a in angles)}],')


def read_coords(mc: MyCobot280, attempts: int = 5):
    for _ in range(attempts):
        c = mc.get_coords()
        if isinstance(c, list) and len(c) >= 6:
            return c
        time.sleep(0.3)
    return None


def pick_place(mc: MyCobot280) -> None:
    print("Capturing the pick-and-place geometry.\n"
          "Move the arm by hand to each position and press Enter.\n")
    captured = {}
    for n, (key, description) in enumerate(PICKPLACE_STEPS, 1):
        print(f"[{n}/{len(PICKPLACE_STEPS)}] {key}")
        print(f"      {description}")
        while True:
            input("      press Enter when posed... ")
            angles, coords = read_angles(mc), read_coords(mc)
            if angles is None or coords is None:
                print("      dropped frame -- try again")
                continue
            captured[key] = (angles, coords)
            print(f"      angles: {[round(a, 1) for a in angles]}")
            print(f"      xyz   : {[round(c, 1) for c in coords[:3]]}\n")
            break

    start_angles, start_coords = captured["start_angles"]
    _, pick_coords = captured["pick"]
    _, place_coords = captured["place"]

    print("\n" + "=" * 70)
    print("Paste into config.py, replacing the PICK_PLACE block:")
    print("=" * 70 + "\n")
    print("PICK_PLACE_TAUGHT = True\n")
    print("PICK_PLACE = {")
    print(f'    "start_angles": [{", ".join(f"{a:.1f}" for a in start_angles)}],')
    print()
    print(f'    "pick_z":   {pick_coords[2]:.0f},')
    print(f'    "place_y":  {place_coords[1]:.0f},')
    print(f'    "place_z":  {place_coords[2]:.0f},')
    print()
    print(f'    "travel_z": None,   # start pose is at z={start_coords[2]:.0f}')
    print("}")
    print("\n" + "=" * 70)

    problems = []
    if pick_coords[2] >= start_coords[2]:
        problems.append(f"pick_z ({pick_coords[2]:.0f}) is not below the start "
                        f"height ({start_coords[2]:.0f}) -- raise the approach pose")
    if place_coords[2] >= start_coords[2]:
        problems.append(f"place_z ({place_coords[2]:.0f}) is not below the start "
                        f"height ({start_coords[2]:.0f})")
    if problems:
        print("\nPROBLEMS -- 'pick' will refuse to run:")
        for p in problems:
            print("  -", p)
    else:
        print("\nGeometry looks consistent. Send 'pick' from WhatsApp -- "
              "with no cube the first time, to watch the path.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickplace", action="store_true",
                    help="guided capture of the four pick-and-place poses")
    args = ap.parse_args()

    mc = connect()
    input("Hold the arm, then press Enter to release the servos... ")
    mc.release_all_servos()
    print()

    try:
        if args.pickplace:
            pick_place(mc)
        else:
            free_form(mc)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        print("Re-engaging servos -- hold the arm.")
        try:
            mc.power_on()
        except Exception:
            pass


if __name__ == "__main__":
    main()
