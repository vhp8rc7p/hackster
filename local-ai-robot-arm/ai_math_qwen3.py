import time
import json
import cv2
import torch
import numpy as np
import re
from PIL import Image
from pymycobot.mycobot280 import MyCobot280

# --- MLX & Transformers Imports ---
from mlx_lm import load as load_llm, generate as generate_llm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# --- Robot Configurations ---
SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
PUMP_LENGTH = 50      
APPROACH_HEIGHT = 100 
PICK_HEIGHT = PUMP_LENGTH + 15 
SPEED = 30
PUMP_PIN = 2
VALVE_PIN = 5
TABLE_Z_BASE = -20.0  

class MathSolverRobot:
    def __init__(self):
        print("Initializing AI Models...")
        
        # 1. Load standard Qwen3 (Text Only)
        print("Loading Qwen 3 1.7B...")
        self.llm, self.tokenizer = load_llm("Qwen/Qwen3-1.7B")
        
        # 2. Load OWL-ViT
        print("Loading OWL-ViT...")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.vit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.vit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        
        # 3. Load Robot Calibration
        print("Loading Hand-Eye Calibration...")
        with open("/Users/v/local-ai-robot-arm/calibration_result.json") as f:
            cal = json.load(f)
        self.K = np.array(cal["camera_matrix"])
        self.dist = np.array(cal["dist_coeffs"])
        self.T_cam2gripper = np.array(cal["T_cam2gripper"])
        
        print("Connecting to myCobot 280...")
        self.mc = MyCobot280(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)
        print("Ready!")

    def capture_image(self):
        print("[📷 Capturing live camera frame...]")
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        for _ in range(5): cap.read() # Warm up
        ret, frame = cap.read()
        cap.release()
        if not ret: raise Exception("Failed to read from camera!")
        
        frame_undistorted = cv2.undistort(frame, self.K, self.dist, None, self.K)
        return frame_undistorted

    def scan_all_symbols(self, frame_cv2):
        print("[👁️ OWL-ViT acting as OCR - Scanning for all numbers and math symbols...]")
        image = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        
        # Ask OWL-ViT to look for every possible digit and math symbol!
        queries = [
            "number 0", "number 1", "number 2", "number 3", "number 4", 
            "number 5", "number 6", "number 7", "number 8", "number 9",
            "plus sign", "minus sign", "equals sign", "multiply sign"
        ]
        
        inputs = self.vit_processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        outputs = self.vit_model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.vit_processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        
        detected_objects = []
        if len(results[0]["boxes"]) > 0:
            for box, score, label_idx in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                label = queries[label_idx]
                
                # Convert OWL-ViT labels to math strings
                math_char = label.replace("number ", "")
                if label == "plus sign": math_char = "+"
                if label == "minus sign": math_char = "-"
                if label == "equals sign": math_char = "="
                if label == "multiply sign": math_char = "*"
                
                detected_objects.append({
                    "char": math_char,
                    "x": cx.item(),
                    "y": cy.item()
                })
        return detected_objects

    def solve_with_qwen(self, equation_string):
        print(f"[🧠 Asking Qwen3 to solve: '{equation_string}']")
        prompt = f"""<|im_start|>system
You are a math solver. Solve the equation provided by the user. Output ONLY the final integer answer. Do not output anything else.<|im_end|>
<|im_start|>user
Equation: {equation_string}<|im_end|>
<|im_start|>assistant
"""
        response = generate_llm(self.llm, self.tokenizer, prompt=prompt, max_tokens=10)
        answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        print(f"[✅ Qwen3 Solved It!]: The answer is {answer}")
        return answer

    def pixel_to_robot_base(self, u, v, tcp_coords):
        x, y, z, rx, ry, rz = tcp_coords
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = Rz @ Ry @ Rx
        T_gripper2base[:3, 3] = [x, y, z]

        T_cam2base = T_gripper2base @ self.T_cam2gripper
        C_base = T_cam2base[:3, 3]
        K_inv = np.linalg.inv(self.K)
        ray_cam = K_inv @ np.array([u, v, 1.0])
        ray_base = T_cam2base[:3, :3] @ ray_cam
        
        if ray_base[2] == 0: return None
        t = (TABLE_Z_BASE - C_base[2]) / ray_base[2]
        return C_base + t * ray_base

    def run(self):
        print("\n--- SETUP ---")
        print("Moving automatically to Observation Pose...")
        # Move to X=150 (forward), Y=0 (center), Z=250 (high up), pointing straight down (-180, 0, 0)
        self.mc.send_coords([150, 0, 250, -180, 0, 0], SPEED)
        time.sleep(4) # Wait for the arm to reach the observation position
        
        tcp = self.mc.get_coords()
        if not tcp: return print("ERROR: Couldn't read robot coordinates.")

        frame = self.capture_image()
        
        # 1. Use OWL-ViT as an OCR to find all symbols
        objects = self.scan_all_symbols(frame)
        if not objects: return print("[❌ Could not see any numbers or symbols!]")
        
        # 2. Sort objects by Y coordinate to separate Equation (bottom) from Candidates (top)
        # You placed candidates "on top of it". Assuming Top means a smaller Y coordinate.
        avg_y = sum(obj["y"] for obj in objects) / len(objects)
        equation_symbols = [obj for obj in objects if obj["y"] > avg_y] # Equation is below
        candidates = [obj for obj in objects if obj["y"] < avg_y]       # Candidates are above
        
        # 3. Sort Equation symbols from Left to Right (X coordinate)
        equation_symbols = sorted(equation_symbols, key=lambda o: o["x"])
        equation_string = "".join(obj["char"] for obj in equation_symbols)
        
        print(f"[🔍 Vision Reconstructed Equation]: {equation_string}")
        
        # 4. Ask Qwen3 to solve the string
        answer = self.solve_with_qwen(equation_string)
        
        # 5. Find the answer among the candidates
        target_candidate = None
        for cand in candidates:
            if cand["char"] == answer:
                target_candidate = cand
                break
                
        if not target_candidate:
            print(f"[❌ I solved it ({answer}), but I don't see that number among the candidates!]")
            return
            
        # 6. Find the Equals Sign to place it behind
        equals_sign = None
        for sym in equation_symbols:
            if sym["char"] == "=":
                equals_sign = sym
                break
                
        if not equals_sign:
            print("[❌ Could not find the equals sign to place the answer behind!]")
            return
            
        # 7. Calculate 3D coordinates
        pick_3d = self.pixel_to_robot_base(target_candidate["x"], target_candidate["y"], tcp)
        # Place it to the right of the equals sign (e.g. + 60 pixels in X)
        place_3d = self.pixel_to_robot_base(equals_sign["x"] + 60, equals_sign["y"], tcp)
        
        print(f"\n[🤖 MOVING ARM] Picking up candidate '{answer}'...")
        self.mc.send_coords([pick_3d[0], pick_3d[1], pick_3d[2] + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        self.mc.send_coords([pick_3d[0], pick_3d[1], pick_3d[2] + PICK_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        self.mc.set_basic_output(VALVE_PIN, 0)
        self.mc.set_basic_output(PUMP_PIN, 0) # GRAB
        time.sleep(2)
        self.mc.send_coords([pick_3d[0], pick_3d[1], pick_3d[2] + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        
        print(f"[🤖 MOVING ARM] Placing '{answer}' behind the '=' sign...")
        self.mc.send_coords([place_3d[0], place_3d[1], place_3d[2] + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        self.mc.send_coords([place_3d[0], place_3d[1], place_3d[2] + PICK_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        self.mc.set_basic_output(PUMP_PIN, 1)
        self.mc.set_basic_output(VALVE_PIN, 1) # RELEASE
        time.sleep(2)
        self.mc.send_coords([place_3d[0], place_3d[1], place_3d[2] + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        
        print("[✅ Task Completed Successfully!]")

if __name__ == "__main__":
    brain = MathSolverRobot()
    brain.run()
