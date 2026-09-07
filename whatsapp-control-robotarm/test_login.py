"""Test the WhatsApp half on its own -- no arm, no serial port, no commands.

    python test_login.py

Prints a QR to scan, confirms the link, then echoes the sender number of every
message it receives so you can copy it into config.ALLOWED_SENDERS.
Ctrl-C to quit. The session is saved, so main.py won't ask you to scan again.
"""

from __future__ import annotations

import argparse
import logging

from neonize.client import NewClient
from neonize.events import (
    ConnectedEv, MessageEv, PairStatusEv, LoggedOutEv, QREv,
    ConnectFailureEv, StreamErrorEv, ClientOutdatedEv, TemporaryBanEv,
)

import config
from qrterm import print_qr
from runner import run_until_interrupt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("login-test")

LOGFILE = "pair-debug.log"

client = NewClient("session.sqlite3")


@client.event(QREv)
def on_qr(_: NewClient, event: QREv):
    for code in event.Codes:
        print("\nScan this in WhatsApp -> Settings -> Linked Devices:\n")
        print_qr(code)
        break       # codes rotate every ~20s; the first is the live one


@client.event(PairStatusEv)
def on_pair(_: NewClient, event: PairStatusEv):
    if event.Error:
        log.error("PAIRING FAILED: %s (status=%s)", event.Error, event.Status)
        log.error("if the phone said 'check internet connection', you probably "
                  "scanned with the camera app -- use WhatsApp -> Settings -> "
                  "Linked Devices -> Link a Device instead")
        return
    log.info("paired! this bot is now linked to +%s", event.ID.User)


@client.event(ConnectFailureEv)
def on_connect_failure(_: NewClient, event: ConnectFailureEv):
    log.error("CONNECT FAILURE reason=%s message=%r", event.Reason, event.Message)


@client.event(StreamErrorEv)
def on_stream_error(_: NewClient, event: StreamErrorEv):
    log.error("STREAM ERROR code=%r", event.Code)


@client.event(ClientOutdatedEv)
def on_outdated(_: NewClient, __: ClientOutdatedEv):
    log.error("CLIENT OUTDATED -- the bundled whatsmeow is older than the "
              "WhatsApp server will accept. A neonize upgrade is the only fix.")


@client.event(TemporaryBanEv)
def on_banned(_: NewClient, event: TemporaryBanEv):
    log.error("TEMPORARY BAN code=%s expires=%s", event.Code, event.Expire)


@client.event(LoggedOutEv)
def on_logged_out(_: NewClient, __: LoggedOutEv):
    log.error("logged out. delete session.sqlite3 and run this again.")


@client.event(ConnectedEv)
def on_connected(cli: NewClient, __: ConnectedEv):
    try:
        me = cli.get_me()
        log.info("connected as +%s (%s)", me.JID.User, me.PushName)
    except Exception:
        log.info("connected.")
    print("\n" + "=" * 62)
    print("  Now message this account from the phone you want to control")
    print("  the arm with. The sender number will appear below.")
    print("=" * 62 + "\n")


@client.event(MessageEv)
def on_message(_: NewClient, event: MessageEv):
    src = event.Info.MessageSource
    msg = event.Message
    text = msg.conversation or msg.extendedTextMessage.text or "<non-text message>"
    ids = [j.User for j in (src.Sender, src.SenderAlt) if getattr(j, "User", "")]
    known = any(i in config.ALLOWED_SENDERS for i in ids)
    mode = "LID" if src.AddressingMode == 2 else "phone number"

    print("-" * 62)
    print(f"  from    : {src.Pushname or '?'}")
    print(f"  ids     : {', '.join(ids) or '?'}   (addressed by {mode})")
    print(f"  chat    : {'group' if src.IsGroup else 'direct'}"
          f"{'  (sent by you)' if src.IsFromMe else ''}")
    print(f"  text    : {text!r}")
    if known:
        print("  status  : WHITELISTED -- main.py will accept this sender")
    else:
        print("  status  : not whitelisted. Add to config.py ALLOWED_SENDERS:")
        print("                ALLOWED_SENDERS = [" +
              ", ".join(f'"{i}"' for i in ids) + "]")
    print("-" * 62 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true",
                    help="capture whatsmeow's own Go-side log to pair-debug.log")
    args = ap.parse_args()

    if args.debug:
        # neonize derives the Go log level from this logger's level
        # (client.py: LogLevel.from_logging(log.level)), so raising it here turns
        # on whatsmeow's internal logging -- including the notification types it
        # receives but doesn't handle.
        logging.getLogger("neonize.client").setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # Debug output goes to the FILE ONLY. Left on the console it interleaves
        # with the QR as it's drawn and shreds it -- the blocks have to be
        # contiguous to scan.
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.INFO)

        handler = logging.FileHandler(LOGFILE, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(handler)
        log.info("debug logging -> %s (file only; console stays clean)", LOGFILE)

    log.info("connecting... a QR will print below if this device isn't linked yet")
    print("(Phone: WhatsApp -> Settings -> Linked Devices -> Link a Device)")
    print("Press Ctrl-C to quit.\n")
    if args.debug:
        print(f"** After scanning, WAIT ~60s before Ctrl-C so the post-scan\n"
              f"** traffic lands in {LOGFILE}.\n")
    run_until_interrupt(client)


if __name__ == "__main__":
    main()
