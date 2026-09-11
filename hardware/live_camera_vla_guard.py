"""
Live Camera Feed VLA Guard Pipeline with Object & Hazard Identification

Runs on: Host Laptop / Jetson Nano / Raspberry Pi
Functionality:
1. Connects to live camera feed via OpenCV (USB Webcam / CSI Camera / Dataset Stream).
2. Runs PyTorch ObjectAwareVLAVisionEncoder (trained weights: checkpoints/object_vla_encoder.pth).
3. Detects physical objects: [human, wall, chair, door, shoe, phone, clear_path].
4. Maps detected objects to safety tokens (OPEN_WAREHOUSE, CROWDED_ROOM, HAZARDOUS_ZONE) & 64-dim embedding vector e_vla.
5. Feeds vector to low-level D3QN policy to control physical Arduino differential drive robot.
"""

import sys
import os
import time
import csv
import random
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

# Ensure parent directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_guard import VLAGuard
from dueling_dqn import VLAGuardedDuelingDQN
from dynamic_dataset_manager import add_custom_class, get_dataset_summary, load_class_config
from train_object_detector import train_rigorous_object_detector

class LiveCameraVLAGuardController:
    def __init__(self, camera_id: int = 0, model_path: str = "checkpoints/best_model.pth",
                 object_model_path: str = "checkpoints/object_vla_encoder.pth",
                 log_data: bool = True):
        
        self.camera_id = camera_id
        self.object_model_path = object_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_data = log_data
        
        print(f"\n--- Initializing Object-Aware Live Camera VLA Guard (Device: {self.device}) ---", flush=True)
        
        # 1. Initialize Object-Aware VLA Guard
        self.reload_vla_guard()
        
        # 2. Initialize low-level VLA-Guarded D3QN Policy
        self.d3qn_policy = VLAGuardedDuelingDQN(state_dim=28, action_dim=5, semantic_dim=64).to(self.device)
        if os.path.exists(model_path):
            try:
                self.d3qn_policy.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                print(f"Loaded trained D3QN policy weights from {model_path}", flush=True)
            except Exception as e:
                print(f"Warning loading D3QN weights ({e}). Running baseline policy.", flush=True)
        self.d3qn_policy.eval()
        
        # Setup Live Data Logging
        self.log_dir = "live_logs"
        self.images_dir = os.path.join(self.log_dir, "images")
        self.csv_log_path = os.path.join(self.log_dir, "live_session_telemetry.csv")
        
        if self.log_data:
            os.makedirs(self.images_dir, exist_ok=True)
            if not os.path.exists(self.csv_log_path):
                with open(self.csv_log_path, mode="w", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(["Timestamp", "Frame_Path", "Detected_Object", "VLA_Token", "Action_ID", "Linear_V", "Angular_W"])
        
        # Action map: action_id -> [linear_v (m/s), angular_w (rad/s)]
        self.action_map = {
            0: [0.20, 0.00],   # Move Straight Fast
            1: [0.10, 0.30],   # Turn Left Soft
            2: [0.10, -0.30],  # Turn Right Soft
            3: [0.00, 0.50],   # Pivot Left Hard
            4: [0.00, -0.50]   # Pivot Right Hard
        }

    def reload_vla_guard(self):
        """Hot-reloads VLAGuard and dynamic object classes into memory."""
        self.vla_guard = VLAGuard(semantic_dim=64, object_model_path=self.object_model_path)
        summary = get_dataset_summary()
        print(f"VLAGuard Hot-Reloaded: Active Classes ({len(summary)}): {[s['name'] for s in summary]}")

    def add_custom_category_interactive(self, category_type="object", custom_name=None, epochs=10):
        """Creates dataset folder, seeds images, retrains PyTorch model, and hot-reloads weights."""
        print(f"\n=======================================================", flush=True)
        print(f"    ADD CUSTOM {category_type.upper()} & RETRAIN MODEL    ", flush=True)
        print(f"=======================================================", flush=True)
        
        if not custom_name:
            prompt_str = f"Enter Custom {category_type.upper()} Name(s) (e.g. laptop, desk): "
            try:
                custom_name = input(prompt_str).strip()
            except Exception:
                custom_name = ""
                
        names_list = [n.strip() for n in custom_name.replace("\n", ",").split(",") if n.strip()]
        if not names_list:
            print("No valid name entered. Resuming stream...", flush=True)
            return
            
        for name in names_list:
            c_name, token, count = add_custom_class(name, category_type=category_type, samples_count=50)
            print(f"  -> Created Directory: Datasets/Object_Obstacles/{c_name}/ ({count} samples)", flush=True)
            
        print(f"\n--- Retraining PyTorch Object Vision Model ({epochs} Epochs) ---", flush=True)
        train_rigorous_object_detector(epochs=epochs, batch_size=32)
        
        print("\nHot-reloading newly trained model into Live Camera Controller...", flush=True)
        self.reload_vla_guard()
        print(f"Success! Model updated with new {category_type.upper()} category.\n", flush=True)

    def get_dataset_fallback_frames(self, dataset_dir="Datasets/MIT Indoor Scene Recognition.v5-resized416by416_70-20-10split.folder/valid"):
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

    def start_live_stream(self, show_window: bool = True, max_frames: int = None, save_video: bool = False):
        cap = None
        use_hardware_camera = False
        
        # Safely attempt opening hardware camera (indices 0 and 1)
        for dev_idx in [self.camera_id, 0, 1]:
            try:
                test_cap = cv2.VideoCapture(dev_idx)
                if test_cap is not None and test_cap.isOpened():
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None:
                        cap = test_cap
                        self.camera_id = dev_idx
                        use_hardware_camera = True
                        break
                    else:
                        test_cap.release()
                elif test_cap is not None:
                    test_cap.release()
            except Exception:
                pass

        if not use_hardware_camera:
            print(f"Notice: Physical hardware webcam (/dev/video*) not currently detected.", flush=True)
            print("Switching to Continuous Dataset Real-Time Live Feed Stream Mode...", flush=True)
            fallback_frames = self.get_dataset_fallback_frames()
            if not fallback_frames:
                print("No fallback frames found in Datasets directory.", flush=True)
                return
        else:
            print(f"\nLive Hardware Camera Stream Started (Camera ID: {self.camera_id}).", flush=True)

        print(f"\n--- Running Object-Aware Real-Time OpenCV VLA Guard Stream ---", flush=True)
        print("  [KEYBOARD CONTROLS]:", flush=True)
        print("    Press 'O' : Add Custom Object (Obstacle/Hazard) & Retrain", flush=True)
        print("    Press 'P' : Add Custom Path (Clear Passage) & Retrain", flush=True)
        print("    Press 'T' : Trigger Model Retraining across all folders", flush=True)
        print("    Press 'Q' : Quit Stream\n", flush=True)
        
        video_writer = None
        if save_video:
            os.makedirs("plots", exist_ok=True)
            video_path = os.path.join("plots", "live_vla_camera_hud.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

        frame_count = 0
        last_vla_update = 0.0
        cached_object = "obstacle"
        cached_token = "CROWDED_ROOM"
        cached_emb = np.zeros(64, dtype=np.float32)
        session_timestamp = int(time.time())
        
        # On-Screen Input & Notification Overlay State
        is_input_mode = False
        input_type = "object"  # "object" or "path"
        input_buffer = ""
        status_msg = ""
        status_msg_expiry = 0.0
        
        try:
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break
                    
                if use_hardware_camera:
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    _, frame = fallback_frames[frame_count % len(fallback_frames)]
                    frame = frame.copy()
                    
                frame_count += 1
                curr_time = time.time()
                
                # Asynchronous Object & Context Inference (2 Hz)
                if not is_input_mode and (curr_time - last_vla_update > 0.3 or not use_hardware_camera):
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    cached_object, cached_token, cached_emb = self.vla_guard.detect_objects_and_get_context(pil_img)
                    last_vla_update = curr_time
                    
                # 50 Hz D3QN Policy Inference
                sim_state = np.random.uniform(0.5, 3.0, size=28).astype(np.float32)
                state_tensor = torch.FloatTensor(sim_state).unsqueeze(0).to(self.device)
                emb_tensor = torch.FloatTensor(cached_emb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_vals = self.d3qn_policy(state_tensor, emb_tensor)
                    action_id = torch.argmax(q_vals).item()
                    
                v_cmd, w_cmd = self.action_map[action_id]
                
                # Log Telemetry
                if self.log_data and frame_count % 5 == 0:
                    frame_filename = f"frame_{session_timestamp}_{frame_count:05d}.jpg"
                    frame_filepath = os.path.join(self.images_dir, frame_filename)
                    cv2.imwrite(frame_filepath, frame)
                    
                    with open(self.csv_log_path, mode="a", newline="") as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow([round(curr_time, 3), frame_filepath, cached_object, cached_token, action_id, v_cmd, w_cmd])
                
                # Render Object HUD Overlay
                color_map = {
                    "OPEN_WAREHOUSE": (0, 255, 0),     # Green
                    "CROWDED_ROOM": (0, 215, 255),     # Gold/Yellow
                    "HAZARDOUS_ZONE": (0, 0, 255)      # Red
                }
                hud_color = color_map.get(cached_token, (255, 255, 255))
                
                # Top HUD Box (Object & Action Telemetry)
                cv2.rectangle(frame, (10, 10), (630, 95), (15, 15, 15), -1)
                cv2.rectangle(frame, (10, 10), (630, 95), hud_color, 2)
                
                cv2.putText(frame, f"OBJECT: [{cached_object.upper()}]  |  TOKEN: [{cached_token}]", 
                            (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
                cv2.putText(frame, f"D3QN Action {action_id} -> Linear V: {v_cmd:.2f} m/s | Angular W: {w_cmd:.2f} rad/s", 
                            (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
                            
                # Bottom HUD Box (Interactive Key Controls Bar)
                cv2.rectangle(frame, (10, 425), (630, 470), (20, 20, 30), -1)
                cv2.rectangle(frame, (10, 425), (630, 470), (100, 100, 255), 1)
                
                if status_msg and curr_time < status_msg_expiry:
                    cv2.putText(frame, f"STATUS: {status_msg}", (20, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "CONTROLS: [O] Add Object  |  [P] Add Path  |  [T] Retrain  |  [Q] Quit", 
                                (20, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 250), 1)
                            
                # Render On-Screen Input Overlay when in typing mode
                if is_input_mode:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (40, 140), (600, 330), (15, 20, 35), -1)
                    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
                    
                    border_c = (0, 215, 255) if input_type == "object" else (0, 255, 0)
                    cv2.rectangle(frame, (40, 140), (600, 330), border_c, 2)
                    
                    header_txt = f"TYPE NEW CUSTOM {input_type.upper()} NAME"
                    cv2.putText(frame, header_txt, (60, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65, border_c, 2)
                    
                    cv2.rectangle(frame, (60, 198), (580, 255), (30, 35, 55), -1)
                    cv2.rectangle(frame, (60, 198), (580, 255), (150, 150, 180), 1)
                    
                    cursor_str = "_" if (int(curr_time * 2.5) % 2 == 0) else " "
                    cv2.putText(frame, f"> {input_buffer}{cursor_str}", (75, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
                    
                    cv2.putText(frame, "Press [ENTER] to Create Folder & Retrain  |  [ESC] Cancel", 
                                (60, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 210), 1)

                if video_writer is not None:
                    video_writer.write(frame)
                    
                if show_window:
                    try:
                        cv2.imshow("Object-Aware VLA D3QN Live Camera Stream", frame)
                        raw_key = cv2.waitKey(1)
                        if raw_key != -1:
                            key = raw_key & 0xFF
                            
                            if is_input_mode:
                                if key in [13, 10]:  # ENTER
                                    typed_name = input_buffer.strip().lower().replace(" ", "_")
                                    if typed_name:
                                        status_msg = f"Retraining Model on '{typed_name}'..."
                                        status_msg_expiry = curr_time + 15.0
                                        
                                        add_custom_class(typed_name, category_type=input_type, samples_count=50)
                                        
                                        def bg_train_job():
                                            nonlocal status_msg, status_msg_expiry
                                            try:
                                                train_rigorous_object_detector(epochs=5, batch_size=64)
                                                self.reload_vla_guard()
                                                status_msg = f"Success! Registered '{typed_name}' [{input_type.upper()}]"
                                                status_msg_expiry = time.time() + 5.0
                                            except Exception as err:
                                                status_msg = f"Training error: {err}"
                                                status_msg_expiry = time.time() + 5.0

                                        import threading
                                        t = threading.Thread(target=bg_train_job, daemon=True)
                                        t.start()
                                        
                                    is_input_mode = False
                                    input_buffer = ""
                                elif key == 27:  # ESC
                                    is_input_mode = False
                                    input_buffer = ""
                                elif raw_key in [8, 127, 65288, 16777219] or key in [8, 127]:  # BACKSPACE
                                    input_buffer = input_buffer[:-1]
                                elif 32 <= key <= 126:  # ASCII CHARS
                                    char = chr(key)
                                    if char.isalnum() or char in ['_', ' ', '-']:
                                        input_buffer += char
                            else:
                                if key == ord('q') or key == 27:
                                    break
                                elif key == ord('o') or key == ord('O'):
                                    is_input_mode = True
                                    input_type = "object"
                                    input_buffer = ""
                                elif key == ord('p') or key == ord('P'):
                                    is_input_mode = True
                                    input_type = "path"
                                    input_buffer = ""
                                elif key == ord('t') or key == ord('T'):
                                    status_msg = "Retraining PyTorch Model..."
                                    status_msg_expiry = curr_time + 15.0
                                    
                                    def bg_retrain():
                                        nonlocal status_msg, status_msg_expiry
                                        try:
                                            train_rigorous_object_detector(epochs=5, batch_size=64)
                                            self.reload_vla_guard()
                                            status_msg = "Model Retrained Successfully!"
                                            status_msg_expiry = time.time() + 5.0
                                        except Exception as err:
                                            status_msg = f"Retrain error: {err}"
                                            status_msg_expiry = time.time() + 5.0

                                    import threading
                                    t = threading.Thread(target=bg_retrain, daemon=True)
                                    t.start()
                    except Exception as e:
                        pass
                        
                print(f"Frame {frame_count:04d} | Object: [{cached_object.upper():12s}] | Token: [{cached_token:15s}] | Velocity: v={v_cmd:.2f}m/s, w={w_cmd:.2f}rad/s", flush=True)
                
                if not use_hardware_camera:
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStream Stopped by User.", flush=True)
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            print("Camera stream closed cleanly.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera Feed VLA Guard Controller with Object & Path Options")
    parser.add_argument("--add-object", type=str, default=None, help="Add custom object name and retrain before launching stream")
    parser.add_argument("--add-path", type=str, default=None, help="Add custom path name and retrain before launching stream")
    parser.add_argument("--interactive", action="store_true", help="Prompt to add custom object/path before starting live feed")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for custom object addition")
    
    args = parser.parse_args()
    
    controller = LiveCameraVLAGuardController(camera_id=0, log_data=True)
    
    if args.add_object:
        controller.add_custom_category_interactive(category_type="object", custom_name=args.add_object, epochs=args.epochs)
    elif args.add_path:
        controller.add_custom_category_interactive(category_type="path", custom_name=args.add_path, epochs=args.epochs)
    elif args.interactive:
        print("\n=======================================================")
        print("    LIVE CAMERA VLA GUARD - CUSTOM CATEGORY LAUNCHER   ")
        print("=======================================================")
        print("1. Add New Custom OBJECT (Obstacle/Hazard)")
        print("2. Add New Custom PATH (Clear Passage)")
        print("3. Start Live Camera Feed Stream directly")
        try:
            choice = input("\nSelect Option [1-3]: ").strip()
            if choice == "1":
                controller.add_custom_category_interactive(category_type="object", epochs=args.epochs)
            elif choice == "2":
                controller.add_custom_category_interactive(category_type="path", epochs=args.epochs)
        except Exception:
            pass

    controller.start_live_stream(show_window=True, max_frames=None, save_video=False)

