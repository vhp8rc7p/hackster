# ==========================================
# 1. Qwen3-1.7B Deployment (via MLX for Mac)
# ==========================================
# Install required package: pip install mlx-lm
from mlx_lm import load, generate

def test_qwen():
    print("Loading Qwen3-1.7B via MLX...")
    model, tokenizer = load("Qwen/Qwen3-1.7B")
    
    prompt = "Write a short poem about a robot learning to see."
    print(f"Prompt: {prompt}")
    
    response = generate(model, tokenizer, prompt=prompt, verbose=True)
    print("Response:", response)

# ==========================================
# 2. OWL-ViT Deployment (via PyTorch MPS)
# ==========================================
# Install required packages: pip install torch torchvision transformers pillow
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import requests

def test_owlvit():
    print("Loading OWL-ViT base patch32...")
    # Use Apple Silicon GPU (MPS) if available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

    # Example image and text queries for zero-shot object detection
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a remote control"]]

    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    print("\nDetection Results:")
    # Target image sizes (height, width) to rescale bounding boxes
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    
    for i in range(len(texts[0])):
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {texts[0][label]} with confidence {round(score.item(), 3)} at location {box}")

if __name__ == "__main__":
    # test_qwen()
    # test_owlvit()
    print("Uncomment the functions above to test the models!")
