import time
import json
import cv2
import torch
import numpy as np
from PIL import Image
from pymycobot.mycobot280 import MyCobot280

# --- MLX & Transformers Imports ---
from mlx_vlm import load as load_vlm, generate as generate_vlm
from mlx_vlm.prompt_utils import get_message_profile
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
TABLE_Z_BASE = -20.0  # Z-height of desktop

class MathSolverRobot:
    def __init__(self):
        print("Initializing AI Models...")
        
        # 1. Load VLM to solve the math equation (Qwen2-VL)
        print("Loading Qwen2-VL (Vision Language Model)...")
        self.vlm_model, self.vlm_processor = load_vlm("mlx-community/Qwen2-VL-2B-Instruct-4bit")
        
        # 2. Load OWL-ViT to find the objects
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
        cv2.imwrite("math_view.png", frame_undistorted)
        return frame_undistorted, "math_view.png"

    def solve_equation_with_vision(self, image_path):
        print("[🧠 Qwen2-VL looking at the equation...]")
        prompt = "Look at the math equation in the image. Solve it. What is the final answer number? Reply with ONLY the digit (e.g. '5')."
        
        # Format the message for Qwen2-VL
        formatted_prompt = self.vlm_processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        response = generate_vlm(self.vlm_model, self.vlm_processor, image_path, formatted_prompt, max_tokens=10)
        answer = response.strip()
        print(f"[✅ Math Solved!]: The answer is {answer}")
        return answer

    def locate_object(self, frame_cv2, target_object):
        print(f"[👁️ OWL-ViT Scanning for '{target_object}']...")
        image = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        inputs = self.vit_processor(text=[[target_object]], images=image, return_tensors="pt").to(self.device)
        outputs = self.vit_model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.vit_processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        
        if len(results[0]["boxes"]) > 0:
            best_idx = results[0]["scores"].argmax()
            box = results[0]["boxes"][best_idx].tolist()
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            print(f"[📍 Found {target_object} at Pixel]: X={cx:.1f}, Y={cy:.1f}")
            return cx, cy
        print(f"[❌ {target_object} not found]")
        return None, None

    def pixel_to_robot_base(self, u, v, tcp_coords):
        """Projects 2D pixel to the 3D table plane using ray intersection"""
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
        self.mc.send_coords([150, 0, 250, -180, 0, 0], SPEED)
        time.sleep(4)
        
        tcp = self.mc.get_coords()
        if not tcp: return print("ERROR: Couldn't read robot coordinates.")

        # 1. Capture the scene
        frame, img_path = self.capture_image()
        
        # 2. VLM solves the math equation
        answer = self.solve_equation_with_vision(img_path)
        
        # 3. Vision finds the correct candidate block
        candidate_target = f"number {answer}"
        cand_px_x, cand_px_y = self.locate_object(frame, candidate_target)
        if cand_px_x is None: return
        
        # 4. Vision finds the equals sign
        eq_px_x, eq_px_y = self.locate_object(frame, "equals sign")
        if eq_px_x is None: return

        # 5. Calculate 3D coordinates
        pick_3d = self.pixel_to_robot_base(cand_px_x, cand_px_y, tcp)
        # Place it slightly to the right of the equals sign (e.g. + 40 pixels in X)
        place_3d = self.pixel_to_robot_base(eq_px_x + 60, eq_px_y, tcp)
        
        print(f"\n[🤖 MOVING ARM] Picking up '{answer}'...")
        self.mc.send_coords([pick_3d[0], pick_3d[1], pick_3d[2] + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        self.mc.send_coords([pick_3d[0], pick_3d[1], pick_3d[2] + PICK_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        self.mc.set_basic_output(VALVE_PIN, 0)
        self.mc.set_basic_output(PUMP_PIN, 0) # GRAB
        time.sleep(2)
        self.mc.send_coords([pick_3d[0], pick_3d[1], pick_3d[2] + APPROACH_HEIGHT, -180, 0, 0], SPEED)
        time.sleep(3)
        
        print(f"[🤖 MOVING ARM] Placing behind the '=' sign...")
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
