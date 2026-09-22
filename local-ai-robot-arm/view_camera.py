#!/usr/bin/env python3
"""
Just open the camera and show the live view.

  ./mlx_env/bin/python view_camera.py          # camera 0
  ./mlx_env/bin/python view_camera.py --camera 1

Keys:  S = save a snapshot   Q or ESC = quit
"""
import argparse
import time

import cv2

ap = argparse.ArgumentParser(description="Live camera view")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
args = ap.parse_args()

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {args.camera}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"camera {args.camera}: {w}x{h}   S = snapshot,  Q = quit")

cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("camera", 1280, 720)

n, t0, fps = 0, time.time(), 0.0
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.02)
        continue

    n += 1
    if n % 10 == 0:
        fps = 10 / (time.time() - t0)
        t0 = time.time()

    disp = frame.copy()
    cv2.putText(disp, f"{w}x{h}   {fps:.0f} fps   S=snapshot  Q=quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("camera", disp)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27):
        break
    if k == ord('s'):
        name = f"snapshot_{int(time.time())}.png"
        cv2.imwrite(name, frame)
        print(f"saved {name}")

cap.release()
cv2.destroyAllWindows()
