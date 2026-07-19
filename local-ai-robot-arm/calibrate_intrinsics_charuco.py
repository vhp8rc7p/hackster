"""
Camera intrinsics calibration using the ChArUco board.

Move the board (or camera) through 15-20 different poses covering the whole
image and varied tilt angles. Each captured frame contributes N corner
correspondences to the intrinsic solver.

Workflow:
  1. Run script. Live preview shows detected ChArUco corners.
  2. SPACE captures the current frame (only if >= 10 corners detected).
  3. Aim for 15-20 captures with different angles/positions.
  4. ENTER solves for K, dist, saves to gantry_calib/intrinsics.json.
     (Old file backed up to gantry_calib/intrinsics.old.json.)

Tips for good coverage:
  - Board in center of image, upper-left, upper-right, lower-left, lower-right
  - Board tilted forward, backward, left, right
  - Board close (fills most of frame) AND far (small in frame)
  - Avoid pure translations without rotation — the solver needs angle variety
"""
import json
import os
import time
import numpy as np
import cv2

SERIAL_PORT = "/dev/tty.usbserial-0202EDB8"
BAUD_RATE = 115200
CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080

# ChArUco config — must match your printed board
BOARD_SQUARES_X = 5
BOARD_SQUARES_Y = 7
SQUARE_MM = 30.0
MARKER_MM = 22.0
DICTIONARY = cv2.aruco.DICT_4X4_50

OUT_DIR = "/Users/v/Downloads/69conference/gantry_calib"
OUT_PATH = os.path.join(OUT_DIR, "intrinsics.json")
BACKUP_PATH = os.path.join(OUT_DIR, "intrinsics.old.json")

MIN_CORNERS_PER_FRAME = 10
MIN_FRAMES = 10


def make_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        SQUARE_MM, MARKER_MM, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    board, detector = make_board()

    print(f"Opening camera id={CAMERA_ID}...")
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(10): cap.read()

    print(f"ChArUco {BOARD_SQUARES_X}x{BOARD_SQUARES_Y}, {SQUARE_MM}mm squares, {MARKER_MM}mm markers")

    # Try releasing arm servos so user can drag the camera around by hand.
    try:
        from pymycobot.mycobot280 import MyCobot280
        mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
        time.sleep(1.5)
        mc.release_all_servos()
        print("Arm servos released — drag the arm to point the camera at varied angles.")
    except Exception as e:
        print(f"(couldn't release arm servos: {e}) — move the board by hand instead")

    print(f"\nAim for {MIN_FRAMES}+ frames covering the full image + varied angles.\n")
    print("Keys: SPACE=capture  U=undo  ENTER=solve  Q=quit\n")

    cv2.namedWindow("intrinsics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("intrinsics", 1280, 720)

    all_obj_points = []
    all_img_points = []
    all_corners_per_frame = []
    image_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02); continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        n_corners = len(ids) if ids is not None else 0

        disp = frame.copy()
        if n_corners > 0:
            for c in corners:
                x, y = int(c[0][0]), int(c[0][1])
                cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
        color = (0, 200, 0) if n_corners >= MIN_CORNERS_PER_FRAME else (0, 165, 255)
        cv2.putText(disp,
                    f"frames: {len(all_obj_points)}   corners: {n_corners}   "
                    f"(need >= {MIN_CORNERS_PER_FRAME})   SPACE=cap  U=undo  ENTER=solve",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("intrinsics", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("Quit without solve."); cap.release(); cv2.destroyAllWindows(); return
        if k == ord('u') and all_obj_points:
            all_obj_points.pop(); all_img_points.pop(); all_corners_per_frame.pop()
            print(f"  ← undo (now {len(all_obj_points)} frames)")
            continue
        if k == 32:  # SPACE
            if n_corners < MIN_CORNERS_PER_FRAME:
                print(f"  ✗ only {n_corners} corners (need ≥ {MIN_CORNERS_PER_FRAME})")
                continue
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            if obj_pts is None or len(obj_pts) < MIN_CORNERS_PER_FRAME:
                print("  ✗ matchImagePoints failed"); continue
            all_obj_points.append(obj_pts)
            all_img_points.append(img_pts)
            all_corners_per_frame.append(n_corners)
            print(f"  ✓ frame {len(all_obj_points):2d}  corners={n_corners}")
            continue
        if k in (13, 10):  # ENTER
            if len(all_obj_points) < MIN_FRAMES:
                print(f"  need ≥ {MIN_FRAMES} frames, have {len(all_obj_points)}")
                continue
            print(f"\nSolving intrinsics from {len(all_obj_points)} frames, "
                  f"{sum(all_corners_per_frame)} total corners...")
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_obj_points, all_img_points, image_size, None, None)
            print(f"\n── Solved ──")
            print(f"  RMS reprojection error: {rms:.3f} px")
            print(f"  fx = {K[0,0]:.2f}   fy = {K[1,1]:.2f}")
            print(f"  cx = {K[0,2]:.2f}   cy = {K[1,2]:.2f}")
            print(f"  dist = {dist.ravel().tolist()}")

            # Backup existing intrinsics
            if os.path.exists(OUT_PATH) and not os.path.exists(BACKUP_PATH):
                with open(OUT_PATH) as f: old = f.read()
                with open(BACKUP_PATH, "w") as f: f.write(old)
                print(f"  Backed up old intrinsics → {BACKUP_PATH}")

            out = {
                "camera_matrix": K.tolist(),
                "dist_coeffs": dist.tolist(),
                "image_size": list(image_size),
                "reprojection_error_px": float(rms),
                "n_frames": len(all_obj_points),
                "board": {
                    "squares_x": BOARD_SQUARES_X, "squares_y": BOARD_SQUARES_Y,
                    "square_mm": SQUARE_MM, "marker_mm": MARKER_MM,
                    "dictionary": "DICT_4X4_50",
                },
                "note": "Camera intrinsics calibrated with ChArUco board.",
            }
            with open(OUT_PATH, "w") as f:
                json.dump(out, f, indent=2)
            print(f"  Wrote {OUT_PATH}")

            if rms > 1.0:
                print(f"  ⚠  RMS > 1 px — recapture with better angle variety")
            elif rms > 0.5:
                print(f"  ✓  RMS acceptable (< 1 px)")
            else:
                print(f"  ✓  Excellent (< 0.5 px)")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
