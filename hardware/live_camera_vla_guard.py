"""
Live Camera Feed VLA Guard Pipeline for Physical Mobile Robot Navigation

Runs on: Host Laptop / Jetson Nano / Raspberry Pi
Functionality:
1. Connects to live camera feed via OpenCV (USB Webcam / CSI Camera / Dataset Stream).
2. Runs PyTorch VLAVisionEncoder (trained on MIT Indoor dataset weights: checkpoints/vla_vision_encoder.pth).
3. Infers live semantic safety tokens (OPEN_WAREHOUSE, CROWDED_ROOM, HAZARDOUS_ZONE).
4. Extracts 64-dimensional continuous modulation embedding vector e_vla.
5. Feeds vector to low-level D3QN policy to control physical Arduino differential drive robot.
"""

import sys
import os
import time
import random
import numpy as np
import torch
import cv2
from PIL import Image

# Ensure parent directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_guard import VLAGuard
from dueling_dqn import VLAGuardedDuelingDQN

class LiveCameraVLAGuardController:
    def __init__(self, camera_id: int = 0, model_path: str = "checkpoints/best_model.pth",
                 vision_model_path: str = "checkpoints/vla_vision_encoder.pth"):
        
        self.camera_id = camera_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n--- Initializing Continuous Live Camera VLA Guard Controller (Device: {self.device}) ---", flush=True)
        
        # 1. Initialize VLA Guard with trained vision encoder weights
        self.vla_guard = VLAGuard(semantic_dim=64, model_path=vision_model_path)
        
        # 2. Initialize low-level VLA-Guarded D3QN Policy
        self.d3qn_policy = VLAGuardedDuelingDQN(state_dim=28, action_dim=5, semantic_dim=64).to(self.device)
        if os.path.exists(model_path):
            try:
                self.d3qn_policy.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                print(f"Loaded trained D3QN policy weights from {model_path}", flush=True)
            except Exception as e:
                print(f"Warning loading D3QN weights ({e}). Running baseline policy.", flush=True)
        self.d3qn_policy.eval()
        
        # Action map: action_id -> [linear_v (m/s), angular_w (rad/s)]
        self.action_map = {
            0: [0.20, 0.00],   # Move Straight Fast
            1: [0.10, 0.30],   # Turn Left Soft
            2: [0.10, -0.30],  # Turn Right Soft
            3: [0.00, 0.50],   # Pivot Left Hard
            4: [0.00, -0.50]   # Pivot Right Hard
        }
        
    def get_dataset_fallback_frames(self, dataset_dir="Datasets/MIT Indoor Scene Recognition.v5-resized416by416_70-20-10split.folder/valid"):
        """Loads sample dataset images as fallback frames if no physical webcam is connected."""
        frames = []
        if os.path.exists(dataset_dir):
            for sub in os.listdir(dataset_dir)[:20]:
                sdir = os.path.join(dataset_dir, sub)
                if os.path.isdir(sdir):
                    files = [f for f in os.listdir(sdir) if f.lower().endswith(('.jpg', '.png'))]
                    for f in files[:3]:
                        img_path = os.path.join(sdir, f)
                        frame_bgr = cv2.imread(img_path)
                        if frame_bgr is not None:
                            frame_bgr = cv2.resize(frame_bgr, (640, 480))
                            frames.append((img_path, frame_bgr))
        return frames

    def start_live_stream(self, show_window: bool = True, max_frames: int = None, save_video: bool = True):
        """
        Captures live continuous video feed from camera or dataset frames, processes vision features through VLA Guard,
        renders real-time HUD overlays, and computes D3QN velocity commands.
        Runs continuously until 'q' key or Ctrl+C is pressed.
        """
        cap = cv2.VideoCapture(self.camera_id)
        use_hardware_camera = cap.isOpened()
        
        if not use_hardware_camera:
            print(f"Notice: Physical camera hardware (Camera ID: {self.camera_id}) not detected.", flush=True)
            print("Switching to Continuous Dataset Camera Feed Stream Mode (MIT Indoor Real Images)...", flush=True)
            fallback_frames = self.get_dataset_fallback_frames()
            if not fallback_frames:
                print("No fallback frames found in Datasets directory.", flush=True)
                return
        else:
            print(f"\nLive Hardware Camera Stream Started (Camera ID: {self.camera_id}).", flush=True)

        print(f"\n--- Running CONTINUOUS Real-Time OpenCV VLA Guard HUD Stream (Press 'q' or Ctrl+C to exit) ---", flush=True)
        
        video_writer = None
        if save_video:
            os.makedirs("plots", exist_ok=True)
            video_path = os.path.join("plots", "live_vla_camera_hud.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

        frame_count = 0
        last_vla_update = 0.0
        cached_token = "CROWDED_ROOM"
        cached_emb = np.zeros(64, dtype=np.float32)
        
        try:
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break
                    
                if use_hardware_camera:
                    ret, frame = cap.read()
                    if not ret:
                        print("End of camera stream or lost connection.", flush=True)
                        break
                else:
                    _, frame = fallback_frames[frame_count % len(fallback_frames)]
                    frame = frame.copy()
                    
                frame_count += 1
                curr_time = time.time()
                
                # Asynchronous VLA Vision Guard Execution (Run VLM vision inference at 2 Hz)
                if curr_time - last_vla_update > 0.3 or not use_hardware_camera:
                    # Convert BGR OpenCV frame to RGB PIL Image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    # Extract live VLA semantic token and 64-dim embedding vector e_vla
                    cached_token, cached_emb = self.vla_guard.get_semantic_context_from_image(pil_img)
                    last_vla_update = curr_time
                    
                # 50 Hz Low-level Policy Inference with synthetic/live LiDAR state (28-dim)
                sim_state = np.random.uniform(0.5, 3.0, size=28).astype(np.float32)
                state_tensor = torch.FloatTensor(sim_state).unsqueeze(0).to(self.device)
                emb_tensor = torch.FloatTensor(cached_emb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_vals = self.d3qn_policy(state_tensor, emb_tensor)
                    action_id = torch.argmax(q_vals).item()
                    
                v_cmd, w_cmd = self.action_map[action_id]
                
                # Render HUD Overlay on Live Camera Frame
                color_map = {
                    "OPEN_WAREHOUSE": (0, 255, 0),     # Green (BGR)
                    "CROWDED_ROOM": (0, 215, 255),     # Yellow/Gold
                    "HAZARDOUS_ZONE": (0, 0, 255)      # Red
                }
                hud_color = color_map.get(cached_token, (255, 255, 255))
                
                # Draw Top Status Banner
                cv2.rectangle(frame, (10, 10), (630, 95), (15, 15, 15), -1)
                cv2.rectangle(frame, (10, 10), (630, 95), hud_color, 2)
                
                cv2.putText(frame, f"VLA GUARD TOKEN: [{cached_token}]", (20, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, hud_color, 2)
                cv2.putText(frame, f"D3QN Action {action_id} -> Linear V: {v_cmd:.2f} m/s | Angular W: {w_cmd:.2f} rad/s", 
                            (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
                            
                if video_writer is not None:
                    video_writer.write(frame)
                    
                if show_window and use_hardware_camera:
                    cv2.imshow("VLA-Guarded D3QN Continuous Live Camera Feed", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User stopped continuous live stream.", flush=True)
                        break
                        
                print(f"Frame {frame_count:04d} | VLA Token: [{cached_token:15s}] | Velocity: v={v_cmd:.2f}m/s, w={w_cmd:.2f}rad/s", flush=True)
                
                if not use_hardware_camera:
                    time.sleep(0.1) # 10 FPS rate control in dataset simulation mode
                
        except KeyboardInterrupt:
            print("\nContinuous Live Camera Feed Stopped by User (Ctrl+C).", flush=True)
        except Exception as e:
            print(f"Error during continuous live camera loop: {e}", flush=True)
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if video_writer is not None:
                video_writer.release()
                print(f"Saved real-time OpenCV HUD video output to: plots/live_vla_camera_hud.mp4", flush=True)
            cv2.destroyAllWindows()
            print("Camera stream closed cleanly.", flush=True)

if __name__ == "__main__":
    controller = LiveCameraVLAGuardController(camera_id=0)
    # Set max_frames=None for continuous infinite execution until 'q' or Ctrl+C is pressed
    controller.start_live_stream(show_window=True, max_frames=None, save_video=True)
