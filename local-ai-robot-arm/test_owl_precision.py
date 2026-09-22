#!/usr/bin/env python3
"""
Measure where the cube-position error comes from.

Two known error sources are separated here:

  A) PARALLAX (affects BOTH hover-test and qwen equally)
     The 2D affine was calibrated by touching markers lying ON THE TABLE.
     A cube is tall, and the camera sees its TOP face. From an overhead
     camera, a point h mm above the table projects outward, away from the
     spot directly under the lens. Feeding that pixel through a table-plane
     affine puts the target too far out by roughly:

         error_mm ≈ (h / D) * r

     h = object height, D = camera height, r = distance from the point
     directly under the camera. This is geometry, not noise — it grows the
     further the cube sits from the image centre.

  B) OWL BBOX BIAS (affects qwen only)
     test_cube_hover.py finds the cube by HSV colour. qwen_command.py finds
     it with OWL, then optionally refines with HSV. If the HSV refinement
     is not firing (wrong colour range for the lighting), the raw OWL box
     centre is used — and OWL boxes include shadow/edge glow, pulling the
     centre off.

This script measures A by prediction and B by direct comparison, so you can
see which one to fix.

USAGE
  ./mlx_env/bin/python test_owl_precision.py                 # green cube
  ./mlx_env/bin/python test_owl_precision.py --color red
  ./mlx_env/bin/python test_owl_precision.py --n 5           # repeat, show jitter
  ./mlx_env/bin/python test_owl_precision.py --save
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

CALIB_PATH = "calibration_affine2d.json"
INTRINSICS = "gantry_calib/intrinsics.json"
FRAME_W, FRAME_H = 1920, 1080
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OWL_THRESHOLD = 0.08

COLOR_RANGES = {
    "green":  [(35, 80, 60), (85, 255, 255)],
    "blue":   [(95, 80, 60), (135, 255, 255)],
    "pink":   [(140, 60, 100), (175, 255, 255)],
    "yellow": [(18, 80, 80), (35, 255, 255)],
    "red":    [(0, 100, 80), (10, 255, 255)],
}


def load_affine():
    with open(CALIB_PATH) as f:
        d = json.load(f)
    return np.array(d["affine_2x3"], dtype=float), d


def px_to_base(A, u, v):
    return A @ np.array([float(u), float(v), 1.0])


def hsv_center(frame, color):
    lo, hi = COLOR_RANGES[color]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 500]
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None
    return (M["m10"]/M["m00"], M["m01"]/M["m00"]), cv2.boundingRect(c)


def owl_center(frame, query, processor, model, device):
    import torch
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=[[query]], images=pil, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
    ms = (time.time() - t0) * 1000
    res = processor.post_process_grounded_object_detection(
        outputs=out, target_sizes=torch.Tensor([pil.size[::-1]]),
        threshold=OWL_THRESHOLD)[0]
    if len(res["boxes"]) == 0:
        return None, None, ms
    i = int(np.argmax(res["scores"].cpu().numpy()))
    b = res["boxes"][i].cpu().numpy()
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0), (b, float(res["scores"][i])), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", default="green", choices=list(COLOR_RANGES))
    ap.add_argument("--n", type=int, default=3, help="repeats (shows jitter)")
    ap.add_argument("--cube-mm", type=float, default=40.0, help="cube height")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(CALIB_PATH):
        sys.exit(f"missing {CALIB_PATH}")
    A, calib = load_affine()

    # Camera height + nadir (the pixel directly under the lens) from intrinsics.
    D = None
    cx_img, cy_img = FRAME_W / 2.0, FRAME_H / 2.0
    if os.path.exists(INTRINSICS):
        K = np.array(json.load(open(INTRINSICS))["camera_matrix"], dtype=float)
        fx, cx_img, cy_img = K[0, 0], K[0, 2], K[1, 2]
        d2 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

    print("=" * 66)
    print("  OWL / AFFINE COORDINATE PRECISION")
    print("=" * 66)
    print(f"  colour: {args.color}   cube height: {args.cube_mm:.0f}mm   "
          f"repeats: {args.n}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8):
        cap.read()

    # Estimate camera height from any visible 25mm ArUco marker.
    ret, f0 = cap.read()
    if os.path.exists(INTRINSICS) and ret:
        c, ids, _ = cv2.aruco.ArucoDetector(
            d2, cv2.aruco.DetectorParameters()).detectMarkers(f0)
        if ids is not None and len(ids):
            sides = [np.mean([np.linalg.norm(cc[0][k]-cc[0][(k+1) % 4])
                              for k in range(4)]) for cc in c]
            D = fx * 25.0 / float(np.mean(sides))
            print(f"  camera height ≈ {D:.0f}mm (from {len(ids)} ArUco markers)")
    if D is None:
        D = 762.0
        print(f"  camera height assumed {D:.0f}mm (no markers visible)")

    print("\n  loading OWL...")
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = Owlv2Processor.from_pretrained(OWL_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(device).eval()
    query = f"a {args.color} cube"

    owl_pts, hsv_pts, deltas, times = [], [], [], []
    print(f"\n  {'#':>2}  {'OWL px':>16} {'HSV px':>16} {'gap px':>7} {'gap mm':>7}")
    print("  " + "-" * 60)
    last = None
    for i in range(args.n):
        for _ in range(3):
            cap.grab()
        ret, frame = cap.read()
        if not ret:
            continue
        last = frame
        o, obox, ms = owl_center(frame, query, processor, model, device)
        h, _ = hsv_center(frame, args.color)
        times.append(ms)
        if o is None or h is None:
            print(f"  {i+1:>2}  "
                  f"{'not found' if o is None else f'{o[0]:7.1f},{o[1]:7.1f}':>16} "
                  f"{'not found' if h is None else f'{h[0]:7.1f},{h[1]:7.1f}':>16}")
            continue
        owl_pts.append(o); hsv_pts.append(h)
        ob = px_to_base(A, *o); hb = px_to_base(A, *h)
        gap_px = float(np.hypot(o[0]-h[0], o[1]-h[1]))
        gap_mm = float(np.linalg.norm(ob - hb))
        deltas.append((gap_px, gap_mm))
        print(f"  {i+1:>2}  {o[0]:7.1f},{o[1]:7.1f} {h[0]:7.1f},{h[1]:7.1f} "
              f"{gap_px:7.1f} {gap_mm:7.1f}")
    cap.release()

    if not deltas:
        print("\n  Could not get both detections — is the cube visible?")
        return 1

    # ── B: OWL vs HSV disagreement ──
    gp = np.array([d[0] for d in deltas]); gm = np.array([d[1] for d in deltas])
    print(f"\n── B) OWL box centre vs HSV centroid " + "─" * 26)
    print(f"  mean gap: {gp.mean():.1f} px  =  {gm.mean():.1f} mm")
    print(f"  (this is the EXTRA error qwen has over the HSV-only hover test)")
    if gm.mean() > 5:
        print(f"  ⚠ OWL's box centre is {gm.mean():.0f}mm off the true colour centre.")
        print(f"    qwen_command refines OWL boxes with HSV — if that refinement")
        print(f"    isn't firing, this is your extra ~1cm. Check CUBE_HSV_RANGES")
        print(f"    matches this cube's real colour.")
    else:
        print(f"  ✓ OWL and HSV agree closely; the extra error is elsewhere.")

    # jitter
    if len(owl_pts) > 1:
        op = np.array(owl_pts); hp = np.array(hsv_pts)
        oj = float(np.max(np.linalg.norm(op - op.mean(axis=0), axis=1)))
        hj = float(np.max(np.linalg.norm(hp - hp.mean(axis=0), axis=1)))
        print(f"\n  frame-to-frame jitter:  OWL {oj:.1f}px   HSV {hj:.1f}px")

    # ── A: parallax ──
    print(f"\n── A) PARALLAX from the cube's height " + "─" * 25)
    u, v = np.mean([p[0] for p in hsv_pts]), np.mean([p[1] for p in hsv_pts])
    r_px = float(np.hypot(u - cx_img, v - cy_img))
    r_mm = r_px * D / fx if os.path.exists(INTRINSICS) else r_px * 0.5
    par = args.cube_mm / D * r_mm
    print(f"  cube is {r_mm:.0f}mm from the point directly under the camera")
    print(f"  predicted outward error = ({args.cube_mm:.0f}/{D:.0f}) x {r_mm:.0f}"
          f"  =  {par:.1f} mm")
    print(f"  → the affine maps the TABLE plane, but you detect the cube TOP,")
    print(f"    so every tall object lands this far OUTWARD from true centre.")
    print(f"    It grows with distance from image centre, and it hits the")
    print(f"    hover test and qwen EQUALLY.")

    print(f"\n── SUMMARY " + "─" * 52)
    print(f"  parallax (both tools)     ≈ {par:5.1f} mm")
    print(f"  OWL bbox bias (qwen only) ≈ {gm.mean():5.1f} mm")
    print(f"  predicted qwen total      ≈ {par + gm.mean():5.1f} mm")
    print(f"  predicted hover-test      ≈ {par:5.1f} mm")
    print(f"\n  OWL inference: {np.median(times):.0f} ms median")
    print("=" * 66)

    if args.save and last is not None:
        disp = last.copy()
        for p, col, lab in ((owl_pts[-1], (0, 128, 255), "OWL"),
                            (hsv_pts[-1], (0, 255, 0), "HSV")):
            cv2.drawMarker(disp, (int(p[0]), int(p[1])), col,
                           cv2.MARKER_CROSS, 40, 3)
            cv2.putText(disp, lab, (int(p[0])+14, int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.imwrite("owl_precision.png", disp)
        print("  saved owl_precision.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
