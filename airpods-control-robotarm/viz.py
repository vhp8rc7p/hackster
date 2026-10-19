#!/usr/bin/env python3
"""
Live 3D view of AirPods head orientation.

Listens for pose packets from `headtrack` on UDP 9871 and relays them to the
browser over Server-Sent Events. The page is served from here so it is
same-origin with the event stream -- no CORS, no websocket handshake, no deps.

    ./headtrack/headtrack        # terminal 1
    ./viz.py                     # terminal 2, then open the printed URL
"""

import json
import os
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

UDP_PORT = 9871
HTTP_PORT = 8770
STREAM_HZ = 40          # above the 25 Hz source, so the view never lags input
HERE = os.path.dirname(os.path.abspath(__file__))

latest = {"pose": None, "stamp": 0.0}


def receive_poses():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", UDP_PORT))
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            latest["pose"] = json.loads(data)
            latest["stamp"] = time.time()
        except (json.JSONDecodeError, OSError):
            continue


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def do_GET(self):
        self.path = urlparse(self.path).path   # drop ?query before routing
        if self.path.startswith("/events"):
            self.stream_events()
        elif self.path in ("/", "/index.html"):
            self.serve_file("head.html", "text/html; charset=utf-8")
        elif self.path.startswith("/models/"):
            self.serve_model(self.path[len("/models/"):])
        else:
            self.send_error(404)

    def serve_model(self, name):
        # No traversal outside models/, and only the types we actually ship.
        if "/" in name or ".." in name:
            return self.send_error(403)
        ctype = {".glb": "model/gltf-binary",
                 ".jpg": "image/jpeg",
                 ".txt": "text/plain; charset=utf-8"}.get(os.path.splitext(name)[1])
        if not ctype:
            return self.send_error(404)
        self.serve_file(os.path.join("models", name), ctype)

    def serve_file(self, name, ctype):
        try:
            with open(os.path.join(HERE, name), "rb") as fh:
                body = fh.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        period = 1.0 / STREAM_HZ
        try:
            while True:
                pose = latest["pose"]
                age = time.time() - latest["stamp"]
                payload = {"live": pose is not None and age < 0.3, "age": round(age, 3)}
                if pose:
                    payload.update(pose)
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser tab closed


def main():
    threading.Thread(target=receive_poses, daemon=True).start()
    url = f"http://127.0.0.1:{HTTP_PORT}/"
    server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler)
    server.daemon_threads = True
    print(f"\n  head view -> {url}")
    print(f"  listening for headtrack on udp :{UDP_PORT}")
    print("  Ctrl-C to stop\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
