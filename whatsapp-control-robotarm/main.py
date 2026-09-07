"""WhatsApp -> myCobot 280 bridge.

Run it, scan the QR with your phone (Settings -> Linked Devices), then text the
arm. The session persists in session.sqlite3, so you only scan once.

    python main.py              # real arm on config.PORT
    python main.py --dry-run    # no serial port, just log what would happen
"""

from __future__ import annotations

import argparse
import logging

from neonize.client import NewClient
from neonize.events import ConnectedEv, MessageEv, PairStatusEv, QREv, LoggedOutEv

import commands
import config
from arm import ArmController
from qrterm import print_qr
from runner import run_until_interrupt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bridge")

SESSION = "session.sqlite3"

client = NewClient(SESSION)
arm: ArmController


def sender_ids(source) -> list[str]:
    """Every identifier this sender could be whitelisted under.

    WhatsApp addresses some senders by LID (a privacy-preserving id like
    123456789012345) rather than by phone number, decided per-chat via
    AddressingMode. The other form, when known, is in SenderAlt. Match on both so
    a whitelist entry works regardless of which mode a given chat uses.
    """
    ids = []
    for jid in (source.Sender, source.SenderAlt):
        user = getattr(jid, "User", "")
        if user and user not in ids:
            ids.append(user)
    return ids


def extract_text(message) -> str:
    """Plain text out of whichever message variant WhatsApp used."""
    if message.conversation:
        return message.conversation
    if message.extendedTextMessage.text:
        return message.extendedTextMessage.text
    return ""


@client.event(QREv)
def on_qr(_: NewClient, event: QREv):
    for code in event.Codes:
        print("\nScan this in WhatsApp -> Settings -> Linked Devices:\n")
        print_qr(code)
        break       # codes rotate; the first is the live one


@client.event(PairStatusEv)
def on_pair(_: NewClient, event: PairStatusEv):
    log.info("paired as %s", event.ID.User)


@client.event(LoggedOutEv)
def on_logged_out(_: NewClient, __: LoggedOutEv):
    log.error("logged out of WhatsApp -- delete %s and re-scan", SESSION)


@client.event(ConnectedEv)
def on_connected(_: NewClient, __: ConnectedEv):
    log.info("connected. allowed senders: %s",
             ", ".join(config.ALLOWED_SENDERS) or "NONE (edit config.py)")


@client.event(MessageEv)
def on_message(cli: NewClient, event: MessageEv):
    source = event.Info.MessageSource

    text = extract_text(event.Message).strip()
    if not text:
        return

    # Our own replies come back as sent-by-you messages. Drop them first, or the
    # bot answers itself in a loop.
    if text.startswith(config.BOT_PREFIX):
        return

    if source.IsFromMe and not config.ALLOW_SELF_COMMANDS:
        return

    ids = sender_ids(source)
    chat = source.Chat
    mode = "LID" if source.AddressingMode == 2 else "PN"
    log.info("msg from %s (%s) [%s]%s: %r", "/".join(ids), event.Info.Pushname,
             mode, " [self]" if source.IsFromMe else "", text)

    def reply(body: str) -> None:
        cli.send_message(chat, config.BOT_PREFIX + body)

    if not any(i in config.ALLOWED_SENDERS for i in ids):
        log.warning("REJECTED %s -- add one of these to ALLOWED_SENDERS: %s",
                    "/".join(ids), ids)
        return                                  # silent: don't confirm the bot exists

    try:
        cmd = commands.parse(text)
    except commands.ParseError as exc:
        reply(f"{exc}\n\nSend 'help' for the list.")
        return

    if cmd is None:
        return                                  # not a command, stay quiet

    if cmd.kind == "help":
        reply(commands.help_text())
        return

    arm.submit(cmd, reply)


def main() -> int:
    global arm

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="don't open the serial port; log moves instead")
    ap.add_argument("--port", default=config.PORT)
    args = ap.parse_args()

    if not config.ALLOWED_SENDERS:
        log.warning("ALLOWED_SENDERS is empty -- every command will be rejected. "
                    "Message the bot once, copy your number from the log, "
                    "and add it to config.py.")

    arm = ArmController(port=args.port, dry_run=args.dry_run)
    try:
        arm.connect()
    except Exception as exc:
        log.error("could not open %s: %s", args.port, exc)
        log.error("is the arm powered on and running the transponder firmware? "
                  "try --dry-run to test the WhatsApp half on its own.")
        return 1

    log.info("connecting to WhatsApp... Ctrl-C to quit.")
    run_until_interrupt(client, on_stop=arm.shutdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
