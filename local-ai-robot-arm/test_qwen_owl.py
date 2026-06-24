"""
Dry-run test: feed instructions to Qwen → JSON plan → OWL detection.
No robot motion. Verifies the Qwen+OWL pipeline end-to-end.
"""
import json, re, time
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from mlx_lm import load as load_llm, generate as generate_llm

CAMERA_ID = 0
FRAME_W, FRAME_H = 1920, 1080
OWL_THRESHOLD = 0.10

QWEN_SYSTEM = """You convert the user's robot command into a strict JSON plan.

Output ONLY a single JSON object with these keys:
  "action": one of ["touch", "home", "quit"]
  "object": a short physical-object description (or null for "home"/"quit")

Examples:
user: "touch the pink cube"              → {"action":"touch","object":"a small pink object"}
user: "place the pump on the yellow one" → {"action":"touch","object":"a small yellow object"}
user: "go to the yellow one"             → {"action":"touch","object":"a small yellow object"}
user: "get the pink one"                 → {"action":"touch","object":"a small pink object"}
user: "place a cube on my hand"          → {"action":"touch","object":"a human hand"}
user: "bring it to my hand"              → {"action":"touch","object":"a human hand"}
user: "go home"                          → {"action":"home","object":null}
user: "back to home"                     → {"action":"home","object":null}
user: "reset"                            → {"action":"home","object":null}
user: "stop"                             → {"action":"quit","object":null}
user: "quit"                             → {"action":"quit","object":null}

Rules:
- A bare color reference ("the yellow one", "the pink one") means a colored cube.
- "reset" means home, not quit.
Do not output any other text."""

INSTRUCTIONS = [
    "touch the pink cube",
    "go to the yellow one",
    "pick up the magenta block",
    "place a cube on my hand",
    "bring it to my hand",
    "drop the cube on my palm",
    "go home",
    "stop",
    "reset",
    "put the pump on the small yellow square",
]


def qwen_plan(llm, tok, user_text):
    # /no_think tells Qwen3 to skip <think> mode → fast direct output
    prompt = (f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{user_text} /no_think<|im_end|>\n"
              f"<|im_start|>assistant\n")
    raw = generate_llm(llm, tok, prompt=prompt, max_tokens=120, verbose=False)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    if not m:
        return None, raw
    try:
        return json.loads(m.group(0)), raw
    except Exception:
        return None, raw


def main():
    print("Loading Qwen3-1.7B...")
    llm, tok = load_llm("Qwen/Qwen3-1.7B")
    print("Loading OWL-ViT...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    print("Capturing one frame for OWL queries...")
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(8): cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("camera read failed"); return
    cv2.imwrite("test_qwen_owl_frame.png", frame)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    print("\n" + "=" * 72)
    annotated = frame.copy()
    colors = [(0,255,0),(0,255,255),(255,0,255),(255,255,0),(0,128,255),
              (128,0,255),(255,128,0),(0,255,128),(128,255,0),(255,0,128)]

    for i, text in enumerate(INSTRUCTIONS):
        print(f"\n[{i+1}/{len(INSTRUCTIONS)}] USER: {text!r}")
        t0 = time.time()
        plan, raw = qwen_plan(llm, tok, text)
        dt = time.time() - t0
        if plan is None:
            print(f"  QWEN: ⚠ no JSON parsed ({dt:.1f}s)")
            print(f"  raw: {raw!r}")
            continue
        print(f"  QWEN: {plan}  ({dt:.1f}s)")

        if plan.get("action") != "touch":
            continue

        obj = plan.get("object")
        if not obj:
            print(f"  OWL : no object"); continue

        inputs = processor(text=[[obj]], images=pil, return_tensors="pt").to(device)
        with torch.no_grad(): outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=torch.Tensor([pil.size[::-1]]),
            threshold=OWL_THRESHOLD)[0]
        if len(res["boxes"]) == 0:
            print(f"  OWL : '{obj}' → not detected")
            continue
        best = max(zip(res["boxes"], res["scores"]), key=lambda z: float(z[1]))
        box = best[0].tolist(); score = float(best[1])
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        print(f"  OWL : '{obj}' → conf={score:.2f}  bbox center=({int(cx)},{int(cy)})")

        color = colors[i % len(colors)]
        x1,y1,x2,y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"#{i+1} {obj}", (x1, max(y1-8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imwrite("test_qwen_owl_annotated.png", annotated)
    print("\n" + "=" * 72)
    print("Saved annotated frame → test_qwen_owl_annotated.png")


if __name__ == "__main__":
    main()
