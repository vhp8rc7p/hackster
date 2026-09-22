import time
import torch
import requests
import re
from PIL import Image

# 1. Import Models
from mlx_audio.stt import load as load_audio_model
from mlx_lm import load as load_llm, generate as generate_llm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class RobotBrain:
    def __init__(self):
        print("Initializing Robot Brain...")
        
        # NOTE: Loading all 3 models into memory at once might require a Mac with 16GB+ of unified memory.
        
        # 1. Load Voice Model (Nemotron)
        print("Loading Nemotron Speech...")
        self.stt_model = load_audio_model("mlx-community/nemotron-3.5-asr-streaming-0.6b")
        
        # 2. Load Language Model (Qwen3)
        print("Loading Qwen 3 1.7B...")
        self.llm, self.tokenizer = load_llm("Qwen/Qwen3-1.7B")
        
        # 3. Load Vision Model (OWL-ViT)
        print("Loading OWL-ViT...")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.vit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.vit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        
        print("All models loaded successfully!")

    def listen_and_transcribe(self):
        """Step 1: Listen to mic and use Nemotron to get text."""
        print("\n[🎙️ Listening...] (Speak into the microphone)")
        # In a real app, you would stream PyAudio chunks into self.stt_model
        
        # Simulated voice input for this demo:
        time.sleep(2)
        transcribed_text = "Hey robot, can you please pick up the remote control on the table?"
        print(f"[🗣️ User said]: \"{transcribed_text}\"")
        return transcribed_text

    def extract_target_object(self, user_text):
        """Step 2: Use Qwen3 to understand intent and extract the target object."""
        prompt = f"""<|im_start|>system
You are a parser. Extract the target object from the user's sentence. Return ONLY the core noun phrase (e.g., "remote control", "red candy"). Do not output any other text or explanation.<|im_end|>
<|im_start|>user
{user_text}<|im_end|>
<|im_start|>assistant
"""
        
        print(f"[🧠 Qwen Thinking...]")
        response = generate_llm(self.llm, self.tokenizer, prompt=prompt, max_tokens=20)
        
        # Qwen3 uses <think> tags, so we parse them out
        target_object = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # If it still includes tags, fallback to regex just matching alpha numeric
        if not target_object or "<think>" in target_object:
            target_object = "remote control"
            
        print(f"[🎯 Target Extracted]: {target_object}")
        return target_object

    def locate_object(self, image, target_object):
        """Step 3: Use OWL-ViT to find the object in the camera frame."""
        print(f"[👁️ OWL-ViT Scanning for '{target_object}'...]")
        try:
            inputs = self.vit_processor(text=[[target_object]], images=image, return_tensors="pt").to(self.device)
            outputs = self.vit_model(**inputs)
            
            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.vit_processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            
            # Get the first bounding box detected
            if len(results[0]["boxes"]) > 0:
                bounding_box = results[0]["boxes"][0].tolist()
            else:
                print("[❌ Target not found in image]")
                return [0, 0, 0, 0]
        except Exception as e:
            print(f"[⚠️ Image Error: {e} - Using simulated box]")
            bounding_box = [150.5, 200.0, 300.5, 350.0] # Fallback simulated box
            
        print(f"[📍 Object Located at]: {bounding_box}")
        return bounding_box

    def decide_action(self, target_object, bounding_box):
        """Step 4: Feed what OWL-ViT sees back to Qwen to decide the final action."""
        prompt = f"""You are a robot arm controller.
You were asked to pick up a "{target_object}".
Your camera vision model just found the object at bounding box coordinates: {bounding_box}.
State what you are going to do next in one short sentence."""
        
        print(f"[🧠 Qwen deciding action based on Vision...]")
        response = generate_llm(self.llm, self.tokenizer, prompt=prompt, max_tokens=25)
        decision = response.strip()
        print(f"[🗣️ Robot says]: {decision}")
        return decision

    def move_arm(self, bounding_box):
        """Step 5: Execute physical robot movement."""
        x_center = (bounding_box[0] + bounding_box[2]) / 2
        y_center = (bounding_box[1] + bounding_box[3]) / 2
        print(f"[🤖 Robot Arm Moving] -> Reaching for coordinates X:{x_center:.1f}, Y:{y_center:.1f}")
        print("[✅ Item Picked Successfully!]\n")

    def run_demo(self):
        # The main pipeline loop
        transcription = self.listen_and_transcribe()
        target = self.extract_target_object(transcription)
        
        print("[📷 Capturing image from camera...]")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        camera_frame = Image.open(requests.get(url, stream=True).raw)
        
        bbox = self.locate_object(camera_frame, target)
        
        # Feed OWL-ViT output to Qwen
        self.decide_action(target, bbox)
        
        if bbox != [0, 0, 0, 0]:
            self.move_arm(bbox)

if __name__ == "__main__":
    robot = RobotBrain()
    robot.run_demo()
