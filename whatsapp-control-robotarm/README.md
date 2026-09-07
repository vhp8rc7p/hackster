# WhatsApp → myCobot 280

Text your robot arm. A Python bridge that logs into WhatsApp as a linked device,
parses incoming messages, and drives an Elephant Robotics myCobot 280 over serial.

```
you: go pickup
bot: moved to 'pickup'
you: down 40
bot: z -40mm -> 62.3
you: grab
bot: gripper -> 10
```

## How it works

| File | Role |
|---|---|
| `main.py` | WhatsApp client, sender whitelist, message → command dispatch |
| `commands.py` | Text parser. Pure functions, no hardware — easy to test |
| `arm.py` | Serial driver + single-threaded work queue |
| `config.py` | Port, allowed senders, joint limits, presets |
| `teach.py` | Pose the arm by hand to capture new presets |

WhatsApp login uses [neonize](https://github.com/krypton-byte/neonize), Python
bindings for `whatsmeow`. You scan a QR once; the session lives in
`session.sqlite3` and survives restarts.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
```

Everything installs into `.venv`, so **activate it first** — a bare `python`
uses your system interpreter and fails with `ModuleNotFoundError: No module
named 'segno'`. If activation is blocked by execution policy, skip it and call
the interpreter directly: `.\.venv\Scripts\python.exe main.py`.

Plug in the arm and confirm the port:

```bash
python -c "import serial.tools.list_ports as l; [print(p) for p in l.comports()]"
```

The M5Stack shows up as a **CH9102** device. Put that port in `config.py`
(currently `COM10`).

On the arm itself, load **Transponder** from the M5Stack's on-screen menu —
without it the arm won't answer serial commands.

### First run

```bash
python main.py --dry-run
```

Dry-run skips the serial port entirely, so you can test the WhatsApp half alone.
A QR code prints in the terminal — scan it in WhatsApp under
**Settings → Linked Devices → Link a Device**.

Now message the bot from your phone. It will ignore you, and the console will log:

```
WARNING bridge: REJECTED 628123456789 -- not in ALLOWED_SENDERS
```

Copy that number into `ALLOWED_SENDERS` in `config.py`, restart, and you're live.
Then drop `--dry-run` to move the real arm.

## Commands

| Command | Effect |
|---|---|
| `home`, `go pickup` | Move to a named pose from `config.PRESETS` |
| `dance`, `wave`, `nod` | Play a routine from `config.ROUTINES`. `do dance` also works |
| `pick` | Pick-and-place: approach, descend, grip, lift, travel, descend, release, retreat. Requires taught poses (below) |
| `up` / `down` / `left` / `right` / `forward` / `back` `[mm]` | Relative move, default 30mm |
| `move <x> <y> <z>` | Absolute coordinates |
| `angles <j1..j6>` | Absolute joint angles |
| `grip open` / `grip close` / `grip 0-100` | Gripper. `grab` and `drop` also work |
| `where` | Current angles, coords, speed |
| `speed <1-70>` | Movement speed |
| `relax` | Release servos — **the arm goes limp and will drop** |
| `stop` | Discard queued moves, and abort a running routine between steps |
| `help` | List the above |

Anything unrecognised is ignored silently, so the bot is usable in a chat where
you also talk normally.

### Adding presets

```bash
python teach.py
```

Releases the servos, then prints the joint angles each time you press Enter.
Pose the arm by hand, press Enter, paste the printed line into `config.PRESETS`.

### Teaching pick-and-place

```bash
python teach.py --pickplace
```

Walks you through four poses — above the cube, at the cube, above the
destination, at the destination — and prints a `PICK_PLACE` block to paste into
`config.py`. Set `PICK_PLACE_TAUGHT = True` at the same time.

Until you do, `pick` refuses to run: the shipped coordinates are invented
placeholders, and driving to them would put the gripper somewhere arbitrary.

First run it **without a cube**, so you can watch the path before anything is
holding something. `stop` aborts between steps, though note the cube stays
gripped if you stop while it's holding one.

## Safety

Both of these are enforced in code, not just documented:

- **Whitelist is fail-closed.** An empty `ALLOWED_SENDERS` rejects everyone.
  Rejections are silent — an unknown sender gets no reply at all, so the bot
  doesn't advertise itself to a wrong number.
- **Limits are clamped.** Joint angles are checked against the 280's real limits
  (`config.JOINT_LIMITS`), speed is capped at `MAX_SPEED`, and a single relative
  move can't exceed `MAX_STEP_MM`.

A few things worth knowing before you leave this running:

- **`relax` drops the arm.** No soft landing. Clear the workspace.
- **One worker thread owns the serial port.** neonize dispatches each message on
  its own thread and pyserial isn't safe to share, so commands queue up and run
  in order. A long move can't block message reception.
- **Group chats work** — the whitelist checks the *sender*, not the chat — but
  anyone in a group with a whitelisted member can't trigger anything, only the
  whitelisted number itself can.

## Ctrl-C

Both scripts run the neonize client on a daemon thread (`runner.py`) so Ctrl-C
works. Calling `client.connect()` directly on the main thread makes the process
**unkillable with Ctrl-C**: it blocks inside a cgo call, the interpreter never
executes bytecode, and the SIGINT handler can't run. A `try/except
KeyboardInterrupt` around it looks right and does nothing.

If a stray process is still stuck from an older build:

```powershell
Get-Process python | Stop-Process -Force
```

Ctrl-Break also works where Ctrl-C doesn't.

## Scan from inside WhatsApp, not with the camera app

**This is the one thing that will waste your afternoon.** The QR encodes a URL
(`https://wa.me/settings/linked_devices#2@...`). Point your phone's **camera app**
at it and the OS opens that link, starting a different linking flow that
whatsmeow doesn't implement — the phone then says *"Couldn't link device"* /
*"check internet connection"* and asks you to scan again, forever.

Scan it from **WhatsApp → Settings → Linked Devices → Link a Device** instead.
That path pairs normally.

Diagnosing it looks like this under `test_login.py --debug` (note the ack-and-drop
about five seconds after a *successful* scan):

```
Recv  <notification type="companion_reg_refresh"><companion_reg_refresh/></notification>
Unhandled notification with type companion_reg_refresh
Send  <ack class="notification" type="companion_reg_refresh"/>
```

If you see that, you scanned the wrong way. whatsmeow tracks it as
[#1177](https://github.com/tulir/whatsmeow/issues/1177); the same notification is
discussed in [Baileys #2737](https://github.com/WhiskeySockets/baileys/issues/2737),
where it's framed as a total pairing outage — it isn't, at least not via the
in-app scanner. You scan, the code is accepted, then the
phone shows *"Couldn't link device"* / *"check internet connection"* and asks you
to scan again. This is not a bug in this project or in your network.

Around **2026-07-28**, WhatsApp added a `<notification type='companion_reg_refresh'>`
to the device-linking flow. No open-source client handles it yet: the client acks
and discards it, `pair-success` is never emitted, the QR pool drains, and the
phone reports failure. It is reproduced in both
[Baileys](https://github.com/WhiskeySockets/baileys/issues/2737) and whatsmeow,
which neonize wraps. Pairing by phone-number code is broken too — it returns
`400 bad-request` — so there's no alternative path.

Status when this was written:

- PyPI's newest neonize is `0.4.3.post0` (2026-07-12), which predates the change.
- GitHub has newer tags (`0.4.4`–`0.4.7`, 2026-08-26), but they ship **no
  uploaded assets** — no wheels, no compiled Go binary. The two "Source code
  (zip/tar.gz)" entries on those release pages are auto-generated by GitHub from
  the tag, not build output; `0.4.3.post0` has 23 real wheels for comparison.
  The wheel build was broken, which is what `0.4.7` itself fixes.
- You *can* build `0.4.7` from the source archive, but it won't help: its
  `goneonize/go.mod` pins `whatsmeow v0.0.0-20260821141805`, i.e. Aug 21 —
  after the break, before any fix.
- whatsmeow's `main` had no `companion_reg_refresh` handler as of its
  2026-08-28 commit — that's the layer any real fix must land in, so building
  neonize from source wouldn't help either.

Re-check the last point in one command — if this prints a match, the fix landed:

```bash
curl -sS --ssl-no-revoke \
  https://raw.githubusercontent.com/tulir/whatsmeow/main/notification.go \
  | grep -n companion_reg_refresh
```

Today it prints nothing: the type falls through to whatsmeow's
`Unhandled notification with type %s` debug line. Don't be fooled by
`link_code_companion_reg`, which *is* handled — that's the phone-pairing-code
flow, a different notification.

A neonize release alone isn't enough either; it has to be one built *after* a
whatsmeow fix, with wheels actually attached.

`test_login.py` logs `PairStatusEv.Error`, `ConnectFailureEv`, `StreamErrorEv`
and `ClientOutdatedEv`, so you can confirm which failure you're actually hitting
rather than assuming this one.

**Until it's fixed, use an official transport instead.** `commands.py` and
`arm.py` have no WhatsApp dependency, so only the transport layer changes —
see the Cloud API note below.

## The ToS caveat

neonize logs in as an unofficial linked device. That's against WhatsApp's terms,
and accounts using unofficial clients do occasionally get banned. Fine for a
hobby build; use a spare number if that matters to you. For anything
production-shaped, use the official WhatsApp Cloud API with a webhook instead —
the `commands.py` and `arm.py` halves port over unchanged.
