import cv2
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# --- Configurations ---
CAMERA_ID = 0
THRESHOLD = 0.05

# These are the things OWL-ViT will constantly look for.
# You can change this list to test different words!
SEARCH_QUERIES = [
    "screw", "metal screw", "small screw", "fastener", "hex bolt",
    "usb flash drive", "usb receiver", "mouse",
    "cube", "block", "candy"
]

def main():
    print("Loading OWL-ViT Model...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    print("Model Loaded!")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

    print("\n[🟢 Live Preview Started]")
    print(f"Looking for: {SEARCH_QUERIES}")
    print("Press 'q' in the window to quit, or 'p' to print current detections to the terminal.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to RGB PIL Image for OWL-ViT
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run inference
        inputs = processor(text=[SEARCH_QUERIES], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=THRESHOLD)

        detected_labels = []

        # Draw bounding boxes
        if len(results[0]["boxes"]) > 0:
            for box, score, label_idx in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
                box = [int(i) for i in box.tolist()]
                label = SEARCH_QUERIES[label_idx]
                confidence = round(score.item(), 2)
                
                detected_labels.append(f"{label} ({confidence})")

                # Draw Rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Draw Label Background
                cv2.rectangle(frame, (box[0], box[1] - 25), (box[0] + 150, box[1]), (0, 255, 0), -1)
                
                # Draw Text
                cv2.putText(frame, f"{label}: {confidence}", (box[0] + 5, box[1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Show the frame
        cv2.imshow("OWL-ViT Live Vision", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"[📷 Snapshot Detections]: {detected_labels}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
