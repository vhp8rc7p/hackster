"""Print a QR code that survives a Windows console.

The compact renderer uses half-block glyphs (U+2580 etc), which the default
Windows codepage (GBK/cp936 here) can't encode -- it raises UnicodeEncodeError
mid-draw. Compact matters though: a WhatsApp pairing payload renders ~35 rows
non-compact, which scrolls off screen before you can scan it. So: switch stdout
to UTF-8, and fall back to the ANSI renderer if that isn't possible.
"""

from __future__ import annotations

import os
import sys

import segno


def _force_utf8() -> bool:
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # Python 3.7+
        return True
    except Exception:
        return False


PNG_PATH = "qr.png"


def print_qr(payload: str) -> None:
    qr = segno.make(payload)

    # Always write an image too. The terminal QR can be mangled by anything that
    # writes to stdout while it's being drawn, and an image can be zoomed.
    saved = None
    try:
        qr.save(PNG_PATH, scale=8, border=4)
        saved = os.path.abspath(PNG_PATH)
    except Exception:
        pass

    if _force_utf8():
        try:
            qr.terminal(compact=True)
        except UnicodeEncodeError:
            qr.terminal()
    else:
        # ANSI inverse-video blocks: pure ASCII, any codepage, twice as tall.
        qr.terminal()

    if saved:
        print(f"\nIf the QR above looks broken, open this image instead:\n  {saved}")
