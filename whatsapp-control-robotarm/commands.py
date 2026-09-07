"""Turn a WhatsApp text message into a Command, or explain why it isn't one.

Parsing is deliberately forgiving: people type on phones, with autocorrect,
capitals and stray punctuation. It is also deliberately strict about numbers --
a malformed number becomes an error message, never a silent default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class Command:
    kind: str                       # angles | coords | relative | gripper | preset | query | relax | stop | speed | help
    args: dict = field(default_factory=dict)


class ParseError(Exception):
    """Message looked like a command but wasn't valid. The text is user-facing."""


# Axis keywords -> (coordinate index, sign). myCobot coords are [x, y, z, rx, ry, rz]
# in mm/degrees, with +x forward, +y left, +z up from the base.
_DIRECTIONS = {
    "up":      (2, +1),
    "down":    (2, -1),
    "left":    (1, +1),
    "right":   (1, -1),
    "forward": (0, +1),
    "fwd":     (0, +1),
    "back":    (0, -1),
    "backward": (0, -1),
}

_ALIASES = {
    "opengripper": "grip open",
    "closegripper": "grip close",
    "open": "grip open",
    "close": "grip close",
    "grab": "grip close",
    "release": "grip open",
    "drop": "grip open",
    "pos": "where",
    "position": "where",
    "status": "where",
}


def _number(token: str, what: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ParseError(f"'{token}' isn't a number ({what})")


def parse(text: str) -> Optional[Command]:
    """Return a Command, None if the message isn't addressed to the arm, or raise
    ParseError with a message worth sending back."""

    # Normalise: lowercase, strip punctuation that phones love to add, collapse spaces.
    raw = text.strip().lower()
    raw = re.sub(r"[!?,;]+$", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return None

    raw = _ALIASES.get(raw.replace(" ", ""), raw)
    parts = raw.split()
    verb, rest = parts[0], parts[1:]

    if verb in ("help", "commands", "?"):
        return Command("help")

    if verb in ("where", "report"):
        return Command("query")

    if verb in ("stop", "halt", "cancel", "abort"):
        return Command("stop")

    if verb in ("relax", "loose", "limp"):
        return Command("relax")

    if verb == "speed":
        if not rest:
            raise ParseError("speed needs a number, e.g. 'speed 40'")
        value = int(_number(rest[0], "speed"))
        if not 1 <= value <= 100:
            raise ParseError("speed must be between 1 and 100")
        return Command("speed", {"value": min(value, config.MAX_SPEED)})

    # Go to the grip position and stop there, gripper open. For checking that the
    # arm actually lines up with the cube before trusting the full sequence.
    if raw in ("pickpoint", "pick point", "goto pick", "go pick", "align"):
        if not config.PICK_PLACE_TAUGHT:
            raise ParseError("pick-and-place poses aren't taught yet -- see config.py")
        return Command("pickpoint")

    if verb in ("pick", "pickplace") or raw in ("pick and place", "pick up cube",
                                                "pick cube", "move cube"):
        if not config.PICK_PLACE_TAUGHT:
            raise ParseError(
                "pick-and-place poses aren't taught yet. The defaults are "
                "placeholders and would crash the arm into something. Run "
                "'python teach.py --pickplace' on the host, paste the poses into "
                "config.py, then set PICK_PLACE_TAUGHT = True."
            )
        return Command("pickplace")

    # Routine names used bare: "dance", "wave", "nod".
    if verb in config.ROUTINES and not rest:
        return Command("routine", {"name": verb})

    if verb in ("do", "play", "run"):
        if not rest:
            raise ParseError("do what? try: " + ", ".join(config.ROUTINES))
        name = rest[0]
        if name not in config.ROUTINES:
            raise ParseError(f"no routine '{name}'. known: " + ", ".join(config.ROUTINES))
        return Command("routine", {"name": name})

    # "home" as a bare word, and any preset name used bare ("pickup").
    if verb in config.PRESETS and not rest:
        return Command("preset", {"name": verb})

    if verb == "go":
        if not rest:
            raise ParseError("go where? try: " + ", ".join(config.PRESETS))
        name = rest[0]
        if name not in config.PRESETS:
            raise ParseError(f"no preset '{name}'. known: " + ", ".join(config.PRESETS))
        return Command("preset", {"name": name})

    if verb in ("grip", "gripper"):
        if not rest:
            raise ParseError("grip open, grip close, or grip 0-100")
        arg = rest[0]
        if arg in ("open", "o"):
            return Command("gripper", {"value": config.GRIPPER_OPEN})
        if arg in ("close", "closed", "shut", "c"):
            return Command("gripper", {"value": config.GRIPPER_CLOSED})
        value = int(_number(arg, "gripper 0-100"))
        if not 0 <= value <= 100:
            raise ParseError("gripper value must be 0-100")
        return Command("gripper", {"value": value})

    if verb in ("angles", "joints", "j"):
        if len(rest) != 6:
            raise ParseError(f"angles needs 6 values, got {len(rest)}")
        values = [_number(t, f"joint {i+1}") for i, t in enumerate(rest)]
        for i, (v, (lo, hi)) in enumerate(zip(values, config.JOINT_LIMITS)):
            if not lo <= v <= hi:
                raise ParseError(f"J{i+1}={v} is outside its limit ({lo} to {hi})")
        return Command("angles", {"values": values})

    if verb in ("move", "coords", "goto"):
        if len(rest) not in (3, 6):
            raise ParseError("move needs 3 values (x y z) or 6 (x y z rx ry rz)")
        values = [_number(t, "coordinate") for t in rest]
        if len(values) == 3:
            values += [0.0, 0.0, 0.0]     # keep current-ish wrist orientation flat
        return Command("coords", {"values": values})

    if verb in _DIRECTIONS:
        axis, sign = _DIRECTIONS[verb]
        distance = float(config.STEP_MM)
        if rest:
            distance = _number(rest[0], "distance in mm")
        if distance <= 0:
            raise ParseError("distance must be positive")
        if distance > config.MAX_STEP_MM:
            raise ParseError(f"{distance:g}mm is too far in one go (max {config.MAX_STEP_MM})")
        return Command("relative", {"axis": axis, "delta": sign * distance})

    return None     # not for us -- stay quiet


HELP_TEXT = """myCobot commands:
- home / go <preset> - {presets}
- dance / wave / nod - routines ({routines})
- pick - pick and place the cube
- up / down / left / right / forward / back [mm]
- move <x> <y> <z> - absolute coords
- angles <j1..j6> - absolute joint angles
- grip open | grip close | grip <0-100>
- where - current position
- speed <1-{maxspeed}>
- relax - release servos (arm goes limp!)
- stop - drop queued moves"""


def help_text() -> str:
    return HELP_TEXT.format(
        presets=", ".join(config.PRESETS),
        routines=", ".join(config.ROUTINES),
        maxspeed=config.MAX_SPEED,
    )
