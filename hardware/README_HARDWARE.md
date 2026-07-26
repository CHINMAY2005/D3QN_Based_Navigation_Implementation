# Physical Hardware Deployment Guide: VLA-Guarded D3QN Robot (Arduino + Host SBC/Laptop)

This directory contains the complete production-ready firmware and serial bridge code to deploy the **VLA-Guarded D3QN Navigation System** onto a physical differential drive mobile robot platform using an **Arduino Microcontroller** and a **Host Computer / Single-Board Computer (Raspberry Pi / Jetson Nano / Laptop)**.

---

## 🏗️ Hardware Architecture & Component List

```
┌────────────────────────────────────────────────────────┐
│  Host Computer / SBC (Raspberry Pi / Jetson / Laptop)  │
│  - Runs PyTorch VLAGuardedDuelingDQN Model (50 Hz)    │
│  - Reads 24-Beam RPLiDAR Range Scans                   │
│  - Receives VLA High-Level Semantic Tokens             │
│  - Transmits "V,<linear_v>,W,<angular_w>\n" via USB    │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ USB Serial UART (@ 115200 Baud)
                            ▼
┌────────────────────────────────────────────────────────┐
│  Arduino Microcontroller (UNO / Mega / Nano)           │
│  - Flashed with arduino_robot_firmware.ino             │
│  - Runs Differential Drive Inverse Kinematics          │
│  - Generates PWM Speed & Direction Signal Outputs      │
│  - Enforces Hardware Emergency Stopping                │
└───────────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│ L298N Motor Driver    │       │ HC-SR04 Ultrasonic    │
│ Dual H-Bridge IC      │       │ Hardware Emergency    │
└───────────┬───────────┘       │ Safety Sensor         │
            │                   └───────────────────────┘
     ┌──────┴──────┐
     ▼             ▼
┌─────────┐   ┌─────────┐
│ Left DC │   │ Right DC│
│ Motor   │   │ Motor   │
└─────────┘   └─────────┘
```

### Required Hardware Components:
1. **Microcontroller**: Arduino UNO, Mega 2560, or Nano.
2. **Motor Driver**: L298N Dual H-Bridge Motor Driver Module (or BTS7960 43A High-Power Driver).
3. **Chassis & Motors**: 2-Wheel / 4-Wheel Differential Drive Robot Chassis with 2x DC Geared Motors.
4. **LiDAR Sensor**: RPLiDAR A1M8 / A2M8 360-degree Laser Scanner (connected over USB).
5. **Safety Sensor**: HC-SR04 Ultrasonic Distance Sensor.
6. **Power Supply**: 12V LiPo or 3x 18650 Battery Pack for Motors + 5V Power for Arduino & Raspberry Pi.
7. **Host Computer**: Raspberry Pi 4 / NVIDIA Jetson Nano / Laptop running Python 3.10+ & PyTorch.

---

## 🔌 Wiring Pinout Table

### 1. Arduino Pinout Connections (Arduino UNO to L298N Motor Driver)

| Arduino Pin | L298N Pin | Function / Description |
| :--- | :--- | :--- |
| **Pin 5 (PWM)** | `ENA` | Left Motor Speed PWM Control |
| **Pin 7** | `IN1` | Left Motor Direction Control 1 |
| **Pin 8** | `IN2` | Left Motor Direction Control 2 |
| **Pin 6 (PWM)** | `ENB` | Right Motor Speed PWM Control |
| **Pin 9** | `IN3` | Right Motor Direction Control 1 |
| **Pin 10** | `IN4` | Right Motor Direction Control 2 |
| **GND** | `GND` | Common Ground (Connect Arduino GND to L298N GND & Battery GND) |

### 2. Ultrasonic Safety Sensor (HC-SR04)

| Arduino Pin | HC-SR04 Pin | Function |
| :--- | :--- | :--- |
| **5V** | `VCC` | 5V Power Supply |
| **Pin 11** | `TRIG` | Trigger Pulse Input |
| **Pin 12** | `ECHO` | Echo Pulse Output |
| **GND** | `GND` | Ground |

---

## 🚀 Step-by-Step Hardware Deployment Instructions

### Step 1: Upload Arduino Firmware
1. Connect your Arduino board to your computer via USB.
2. Open `arduino_robot_firmware.ino` in **Arduino IDE**.
3. Select your Board (*Tools -> Board -> Arduino Uno*) and Port (*Tools -> Port*).
4. Click **Upload**.

### Step 2: Install PySerial on Host Computer
On your host laptop or Jetson Nano, install PySerial:
```bash
pip install pyserial
```

### Step 3: Run Host Bridge Controller
Connect the Arduino to your host computer via USB (`/dev/ttyUSB0` or `/dev/ttyACM0`). Execute the bridge script:

```bash
# Dry Run / Test Mode (Serial Emulation):
python3 hardware/bridge_host_controller.py

# Real Hardware Deployment (Serial Connected):
python3 -c "
from hardware.bridge_host_controller import PhysicalRobotBridge
bridge = PhysicalRobotBridge(model_path='checkpoints/best_model.pth', serial_port='/dev/ttyUSB0', use_hardware_serial=True)
bridge.run_control_loop(duration_sec=60.0, goal_x=8.0, goal_y=8.0)
"
```

---

## 🛡️ Built-in Hardware Safety Features
1. **Serial Watchdog Timeout**: If the host computer fails or loses USB connection for $> 500\text{ms}$, the Arduino firmware automatically cuts motor PWM to 0 to prevent runaway crashes.
2. **Ultrasonic Emergency Stop**: If any physical obstacle approaches within $< 15\text{cm}$, the Arduino hardware immediately overrides motor inputs and stops.
3. **PWM Deadband Compensation**: Maps non-linear low-voltage motor friction, preventing motor stalling at low speeds.
