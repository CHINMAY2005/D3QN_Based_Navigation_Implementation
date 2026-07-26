"""
Host Bridge Controller for VLA-Guarded D3QN Physical Robot Deployment

Runs on: Host Laptop / Jetson Nano / Raspberry Pi
Bridge Role:
1. Connects to Arduino over Serial (USB UART @ 115200 baud).
2. Collects 24-beam 360-degree LiDAR range data (RPLiDAR A1/A2 or distance array).
3. Receives high-level VLA semantic tokens & modulation embeddings e_vla (64-dim).
4. Runs real-time PyTorch inference on VLAGuardedDuelingDQN (50 Hz control loop).
5. Transmits velocity primitives [v, w] to Arduino motor firmware.
"""

import sys
import os
import time
import math
import numpy as np
import torch

# Ensure parent directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dueling_dqn import VLAGuardedDuelingDQN
from vla_guard import VLAGuard

try:
    import serial
except ImportError:
    print("PySerial library not found. Install via: pip install pyserial")
    serial = None

class PhysicalRobotBridge:
    def __init__(self, model_path: str = "checkpoints/best_model.pth", 
                 serial_port: str = "/dev/ttyUSB0", baud_rate: int = 115200,
                 use_hardware_serial: bool = False):
        
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.use_hardware_serial = use_hardware_serial
        self.ser = None
        
        # Action space primitive mapping: action_id -> [v (m/s), w (rad/s)]
        self.action_map = {
            0: [0.20, 0.00],   # Move Straight Fast
            1: [0.10, 0.30],   # Turn Left Soft
            2: [0.10, -0.30],  # Turn Right Soft
            3: [0.00, 0.50],   # Pivot Left Hard
            4: [0.00, -0.50]   # Pivot Right Hard
        }
        
        # Initialize VLA Guard & PyTorch D3QN Neural Network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vla_guard = VLAGuard(semantic_dim=64)
        self.model = VLAGuardedDuelingDQN(state_dim=28, action_dim=5, semantic_dim=64).to(self.device)
        
        # Load trained PyTorch checkpoint
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
                print(f"Loaded trained VLA-Guarded D3QN model weights from: {model_path}")
            except Exception as e:
                print(f"Warning loading weights ({e}). Running with initialized model.")
        else:
            print(f"Checkpoint {model_path} not found. Running with baseline model.")
            
        self.model.eval()
        
        # Establish Serial Connection to Arduino
        if self.use_hardware_serial and serial is not None:
            try:
                self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
                time.sleep(2.0)  # Wait for Arduino auto-reset
                print(f"Connected to Arduino hardware on port: {self.serial_port} @ {self.baud_rate} baud.")
            except Exception as e:
                print(f"Failed to open Serial port {self.serial_port}: {e}")
                self.ser = None
        else:
            print("Running in Serial Emulation Mode (Dry Run).")
            
    def get_physical_lidar_scan(self) -> np.ndarray:
        """
        Reads 24 radial beams (360 degrees) from RPLiDAR hardware or range array.
        Returns clamped distances [0.0m, 3.0m].
        """
        # In physical deployment, replace this with RPLiDAR SDK query:
        # scan = rplidar.get_24_beam_scan()
        # Simulated range for dry run testing:
        scan = np.random.uniform(0.8, 3.0, size=24).astype(np.float32)
        return scan

    def send_velocity_command(self, v: float, w: float):
        """
        Formats and transmits velocity command string to Arduino over Serial.
        Protocol: "V,<linear_v>,W,<angular_w>\n"
        """
        cmd_str = f"V,{v:.2f},W,{w:.2f}\n"
        
        if self.ser is not None and self.ser.is_open:
            self.ser.write(cmd_str.encode('utf-8'))
            self.ser.flush()
        else:
            # Emulated Serial output logging
            print(f"[EMULATED SERIAL TX -> ARDUINO]: {cmd_str.strip()}")

    def run_control_loop(self, duration_sec: float = 60.0, goal_x: float = 8.0, goal_y: float = 8.0):
        """
        Executes real-time 50 Hz control loop interfacing sensor perception,
        VLA semantic guarding, PyTorch policy inference, and Arduino transmission.
        """
        print(f"\n--- Starting Real-Time Physical Robot Control Loop (Target Goal: [{goal_x}, {goal_y}]) ---")
        start_time = time.time()
        loop_interval = 0.02  # 50 Hz (20ms per cycle)
        
        # Initial robot estimated pose
        rx, ry, r_theta = 1.0, 1.0, 0.0
        
        try:
            while time.time() - start_time < duration_sec:
                loop_start = time.time()
                
                # 1. Acquire 24-beam LiDAR scan from physical sensor
                lidar_scan = self.get_physical_lidar_scan()
                min_lidar_dist = np.min(lidar_scan)
                
                # 2. Compute goal tracking features
                dx = goal_x - rx
                dy = goal_y - ry
                dist_to_goal = math.sqrt(dx**2 + dy**2)
                goal_heading = math.atan2(dy, dx)
                heading_error = (goal_heading - r_theta + math.pi) % (2 * math.pi) - math.pi
                
                # Assemble 28-dimensional state vector
                state_vec = np.zeros(28, dtype=np.float32)
                state_vec[:24] = lidar_scan
                state_vec[24] = dist_to_goal
                state_vec[25] = heading_error
                state_vec[26] = 0.0  # current v
                state_vec[27] = 0.0  # current w
                
                # 3. Query High-Level VLA Guard for semantic context & modulation vector
                vla_token, sem_emb = self.vla_guard.get_semantic_context(rx, ry, min_lidar_dist)
                
                # 4. Run PyTorch Neural Network Inference
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
                sem_tensor = torch.FloatTensor(sem_emb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_values = self.model(state_tensor, sem_tensor)
                    action_id = torch.argmax(q_values).item()
                    
                v_cmd, w_cmd = self.action_map[action_id]
                
                # 5. Transmit command to Arduino motor controller
                self.send_velocity_command(v_cmd, w_cmd)
                
                # Log telemetry output
                print(f"Step | Goal Dist: {dist_to_goal:.2f}m | VLA Token: [{vla_token}] | Action {action_id} -> v={v_cmd:.2f}m/s, w={w_cmd:.2f}rad/s")
                
                # Check goal arrival condition
                if dist_to_goal < 0.3:
                    print("GOAL ARRIVAL REACHED! Sending STOP command to Arduino.")
                    self.send_velocity_command(0.0, 0.0)
                    break
                    
                # Maintain 50 Hz loop frequency
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, loop_interval - elapsed))
                
        except KeyboardInterrupt:
            print("\nEmergency Stop Triggered by User!")
        finally:
            self.send_velocity_command(0.0, 0.0)
            if self.ser is not None:
                self.ser.close()
            print("Control loop terminated cleanly.")

if __name__ == "__main__":
    # To run with real physical Arduino plugged in: set use_hardware_serial=True
    bridge = PhysicalRobotBridge(model_path="checkpoints/best_model.pth", 
                                  serial_port="/dev/ttyUSB0", 
                                  use_hardware_serial=False)
    bridge.run_control_loop(duration_sec=15.0)
