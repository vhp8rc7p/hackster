import time
import json
import cv2
import torch
import requests
import re
import numpy as np
from PIL import Image
from pymycobot.mycobot280 import MyCobot280

# --- MLX & Transformers Imports ---
from mlx_audio.stt import load as load_audio_model
from mlx_lm import load as load_llm, generate as generate_llm
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# --- Robot Configurations ---
SERIAL_PORT = "/dev/tty.usbserial-5AE20107941"
BAUD_RATE = 115200
CAMERA_ID = 0
PUMP_LENGTH = 50      # mm
APPROACH_HEIGHT = 100 # mm
PICK_HEIGHT = PUMP_LENGTH + 15 
SPEED = 30
PUMP_PIN = 2
VALVE_PIN = 5

# Assuming the objects sit on the table, you need to measure the table's Z-height in the robot base frame
TABLE_Z_BASE = -20.0 # <--- YOU MIGHT NEED TO TWEAK THIS Z HEIGHT FOR YOUR TABLE

class RobotAIBrain:
    def __init__(self):
        print("Initializing AI Models...")
        self.stt_model = load_audio_model("mlx-community/nemotron-3.5-asr-streaming-0.6b")
        self.llm, self.tokenizer = load_llm("Qwen/Qwen3-1.7B")
        
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.vit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.vit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        
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

    def listen_and_transcribe(self):
        print("\n[🎙️ Simulated Mic]: 'Hey robot, please pick up the red cube.'")
        # In the future, hook this up to PyAudio streaming!
        return "Hey robot, please pick up the red cube."

    def extract_target_object(self, text):
        prompt = f"<|im_start|>system\nYou extract the target object from the user's sentence. Return ONLY the core noun phrase (e.g., 'red candy', 'red cube'). Do not output any other text.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        response = generate_llm(self.llm, self.tokenizer, prompt=prompt, max_tokens=20)
        target = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        if not target or "<think>" in target: target = "red cube"
        print(f"[🎯 Qwen Intent]: Wants to pick up -> {target}")
        return target

    def capture_image(self):
        print("[📷 Capturing live camera frame...]")
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        for _ in range(5): cap.read() # Warm up
        ret, frame = cap.read()
        cap.release()
        if not ret: raise Exception("Failed to read from camera!")
        
        # Undistort image using calibration
        frame_undistorted = cv2.undistort(frame, self.K, self.dist, None, self.K)
        cv2.imwrite("ai_view.png", frame_undistorted)
        return frame_undistorted

    def locate_object(self, frame_cv2, target_object):
        print(f"[👁️ OWL-ViT Scanning for '{target_object}']...")
        image = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        inputs = self.vit_processor(text=[[target_object]], images=image, return_tensors="pt").to(self.device)
        outputs = self.vit_model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.vit_processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        
        if len(results[0]["boxes"]) > 0:
            best_idx = results[0]["scores"].argmax()
            box = results[0]["boxes"][best_idx].tolist() # [xmin, ymin, xmax, ymax]
            # Get pixel center
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            print(f"[📍 Found at Pixel]: X={cx:.1f}, Y={cy:.1f}")
            return cx, cy
        print("[❌ Object not found]")
        return None, None

    def pixel_to_robot_base(self, u, v, tcp_coords):
        """Projects 2D pixel to the 3D table plane using ray intersection"""
        # 1. Get T_gripper2base
        x, y, z, rx, ry, rz = tcp_coords
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
        Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
        Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
        Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = Rz @ Ry @ Rx
        T_gripper2base[:3, 3] = [x, y, z]

        # 2. Get T_cam2base
        T_cam2base = T_gripper2base @ self.T_cam2gripper
        
        # 3. Camera origin in base frame
        C_base = T_cam2base[:3, 3]

        # 4. Create 3D ray from pixel in camera frame
        K_inv = np.linalg.inv(self.K)
        ray_cam = K_inv @ np.array([u, v, 1.0])
        
        # 5. Transform ray to base frame
        ray_base = T_cam2base[:3, :3] @ ray_cam
        
        # 6. Intersect ray with Z=TABLE_Z_BASE plane
        # P = C_base + t * ray_base -> P_z = C_base_z + t * ray_base_z = TABLE_Z_BASE
        if ray_base[2] == 0: return None # Parallel to table
        t = (TABLE_Z_BASE - C_base[2]) / ray_base[2]
        
        P_base = C_base + t * ray_base
        return P_base

    def run(self):
        text = self.listen_and_transcribe()
        target = self.extract_target_object(text)
        
        # Read current robot TCP to establish where the camera is looking from
        tcp = self.mc.get_coords()
        if not tcp:
            print("ERROR: Couldn't read robot coordinates.")
            return

        frame = self.capture_image()
        cx, cy = self.locate_object(frame, target)
        
        if cx is None: return

        # Transform Pixel -> Physical 3D Coordinate
        target_3d = self.pixel_to_robot_base(cx, cy, tcp)
        print(f"[🗺️ 3D Calculation]: Target is at Base X:{target_3d[0]:.1f}, Y:{target_3d[1]:.1f}, Z:{target_3d[2]:.1f}")
        
        # Physical Robot Movement
        pick_x, pick_y, pick_z = target_3d[0], target_3d[1], target_3d[2]
        
        print(f"\n[🤖 MOVING ARM] Approaching {target}...")
        self.mc.send_coords([pick_x, pick_y, pick_z + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        
        print("[🤖 Lowering & Gripping]...")
        self.mc.send_coords([pick_x, pick_y, pick_z + PICK_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        self.mc.set_basic_output(VALVE_PIN, 0)
        self.mc.set_basic_output(PUMP_PIN, 0)
        time.sleep(2)
        
        print("[🤖 Lifting]...")
        self.mc.send_coords([pick_x, pick_y, pick_z + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        print("[✅ Item Picked Successfully!]")

if __name__ == "__main__":
    brain = RobotAIBrain()
    brain.run()
