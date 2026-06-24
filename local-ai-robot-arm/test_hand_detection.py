"""
Standalone hand detection test. Shows ALL OWL detections for hand-related
queries with their confidence. Use this to find what threshold reliably
separates your real hand from false positives.

Usage:
  python test_hand_detection.py
    - Press SPACE to capture & analyze the current frame in detail
    - Press 1/2/3 to toggle which queries are active
    - Press 'q' to quit
"""
import time
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080

# Try these queries to see which one matches your hand best
QUERIES = [
    "a human hand",
    "a hand",
    "an open palm",
]

# Show every detection above this very low threshold so we can see noise too
SHOW_ABOVE = 0.02


def main():
    print("Loading OWL-ViT...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    print(f"  device={device}")

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(5): cap.read()

    print(f"\nLive detection. Queries: {QUERIES}")
    print("  SPACE = print full detection list to terminal")
    print("  q = quit\n")

    last_owl = 0
    last_dets = []  # list of (query, box, score)

    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.05); continue
        now = time.time()

        # Run OWL every ~400ms so the window stays smooth
        if now - last_owl > 0.4:
            last_owl = now
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(text=[QUERIES], images=pil, return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            owl_ms = (time.time() - t0) * 1000

            res = processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=torch.Tensor([pil.size[::-1]]),
                threshold=SHOW_ABOVE,
            )[0]

            last_dets = []
            for box, score, label_idx in zip(res["boxes"], res["scores"], res["labels"]):
                q = QUERIES[int(label_idx)]
                last_dets.append((q, [int(v) for v in box.tolist()], float(score)))

            # Print top 3 per query so terminal shows live scores
            print(f"\n--- OWL inference: {owl_ms:.0f}ms ---")
            for q in QUERIES:
                hits = sorted([d for d in last_dets if d[0] == q],
                              key=lambda d: -d[2])[:3]
                if hits:
                    s = "  ".join(f"{d[2]:.3f}@({(d[1][0]+d[1][2])//2},{(d[1][1]+d[1][3])//2})"
                                  for d in hits)
                    print(f"  {q:18s} → {s}")
                else:
                    print(f"  {q:18s} → no detections")

        # Draw all detections on the live frame, color by query
        disp = frame.copy()
        colors = {"a human hand": (0, 255, 0),     # green
                  "a hand":       (0, 255, 255),   # yellow
                  "an open palm": (255, 0, 255)}   # magenta
        for q, bb, sc in last_dets:
            color = colors.get(q, (255, 255, 255))
            cv2.rectangle(disp, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
            cv2.putText(disp, f"{q[:14]} {sc:.2f}",
                        (bb[0], max(bb[1] - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Legend
        y = 30
        for q, c in colors.items():
            cv2.putText(disp, q, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            y += 25
        cv2.putText(disp, "SPACE = full dump, q = quit",
                    (20, disp.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("hand_detection_test", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord(' '):
            print("\n=== FULL DETECTION DUMP ===")
            for q, bb, sc in sorted(last_dets, key=lambda d: -d[2]):
                print(f"  {q:18s}  conf={sc:.3f}  bbox=({bb[0]},{bb[1]})-({bb[2]},{bb[3]})")
            print("===========================")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
