/*
 * Arduino Differential Drive Motor Firmware for VLA-Guarded D3QN Robot
 * 
 * Hardware Architecture:
 * - Microcontroller: Arduino UNO / Mega / Nano
 * - Motor Driver: L298N Dual H-Bridge Motor Driver
 * - Communication: Serial USB (UART) at 115200 Baud from Host SBC/Laptop
 * - Safety: Ultrasonic Emergency Stop Backup (HC-SR04) + Serial Timeout watchdog
 * 
 * Command Protocol:
 * "V,<linear_v>,W,<angular_w>\n"
 *   Where:
 *     <linear_v>  = translational velocity in m/s (e.g., 0.20)
 *     <angular_w> = rotational velocity in rad/s (e.g., 0.30 or -0.50)
 */

#include <Arduino.h>

// =====================================================================
// PIN DEFINITIONS (L298N Motor Driver & Sensors)
// =====================================================================
// Left Motor Pins
const int ENA_PIN = 5;   // PWM Speed Pin Left Motor (Timer 0/2 PWM)
const int IN1_PIN = 7;   // Direction Pin 1 Left Motor
const int IN2_PIN = 8;   // Direction Pin 2 Left Motor

// Right Motor Pins
const int ENB_PIN = 6;   // PWM Speed Pin Right Motor (Timer 0/2 PWM)
const int IN3_PIN = 9;   // Direction Pin 1 Right Motor
const int IN4_PIN = 10;  // Direction Pin 2 Right Motor

// Hardware Safety Pins (HC-SR04 Ultrasonic Backup)
const int TRIG_PIN = 11;
const int ECHO_PIN = 12;

// =====================================================================
// PHYSICAL ROBOT SPECIFICATIONS & CONSTANTS
// =====================================================================
const float WHEEL_BASE = 0.20;     // Distance between left and right wheels (meters, L = 20cm)
const float MAX_SPEED_MS = 0.50;   // Maximum linear speed physical motors can produce (m/s)
const int MIN_PWM = 40;            // Minimum PWM threshold to overcome static friction (Deadband)
const int MAX_PWM = 255;           // Maximum PWM output value

// Watchdog & Safety Constants
const unsigned long SERIAL_TIMEOUT_MS = 500; // Emergency stop if serial drops for > 500ms
const float EMERGENCY_STOP_DIST_CM = 15.0;  // Emergency hardware brake threshold in cm

// =====================================================================
// STATE VARIABLES
// =====================================================================
unsigned long lastSerialTime = 0;
float currentLinearV = 0.0;
float currentAngularW = 0.0;

// =====================================================================
// FUNCTION DECLARATIONS
// =====================================================================
void setMotorSpeeds(float v_m_s, float w_rad_s);
void driveMotorLeft(int pwmSpeed);
void driveMotorRight(int pwmSpeed);
void stopMotors();
float readUltrasonicDistanceCM();
void parseSerialCommand(String cmd);

void setup() {
  // Initialize Serial Communication at 115200 baud
  Serial.begin(115200);
  
  // Set Motor Control Pins as Outputs
  pinMode(ENA_PIN, OUTPUT);
  pinMode(ENB_PIN, OUTPUT);
  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);
  pinMode(IN3_PIN, OUTPUT);
  pinMode(IN4_PIN, OUTPUT);
  
  // Set Ultrasonic Safety Pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // Initially stop all motors
  stopMotors();
  lastSerialTime = millis();
  
  Serial.println("VLA-Guarded D3QN Arduino Motor Controller Ready.");
}

void loop() {
  // 1. Check Serial Watchdog (Safety Timeout)
  if (millis() - lastSerialTime > SERIAL_TIMEOUT_MS) {
    stopMotors();
  }

  // 2. Hardware Ultrasonic Safety Backup Check
  float obsDistCM = readUltrasonicDistanceCM();
  if (obsDistCM > 0.0 && obsDistCM < EMERGENCY_STOP_DIST_CM && currentLinearV > 0.0) {
    stopMotors();
    Serial.println("SAFETY_BRAKE: Obstacle detected within emergency threshold!");
  }

  // 3. Read & Parse Incoming Commands from Host PyTorch Controller
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');
    inputString.trim();
    if (inputString.length() > 0) {
      parseSerialCommand(inputString);
      lastSerialTime = millis();
    }
  }
}

// =====================================================================
// DIFFERENTIAL DRIVE INVERSE KINEMATICS & MOTOR CONTROL
// =====================================================================

void parseSerialCommand(String cmd) {
  // Expected format: "V,0.20,W,0.30" or "STOP"
  if (cmd.equals("STOP")) {
    stopMotors();
    return;
  }

  if (cmd.startsWith("V,")) {
    int wIndex = cmd.indexOf(",W,");
    if (wIndex != -1) {
      String vStr = cmd.substring(2, wIndex);
      String wStr = cmd.substring(wIndex + 3);
      
      currentLinearV = vStr.toFloat();
      currentAngularW = wStr.toFloat();
      
      setMotorSpeeds(currentLinearV, currentAngularW);
    }
  }
}

void setMotorSpeeds(float v, float w) {
  // Inverse Kinematics for Differential Drive:
  // v_left  = v - (w * WHEEL_BASE / 2)
  // v_right = v + (w * WHEEL_BASE / 2)
  float vLeft  = v - (w * WHEEL_BASE / 2.0);
  float vRight = v + (w * WHEEL_BASE / 2.0);

  // Map velocity (m/s) to PWM range (-255 to 255)
  int pwmLeft  = (int)((vLeft  / MAX_SPEED_MS) * 255.0);
  int pwmRight = (int)((vRight / MAX_SPEED_MS) * 255.0);

  // Apply PWM Deadband Compensation & Clamping
  pwmLeft  = constrain(pwmLeft,  -MAX_PWM, MAX_PWM);
  pwmRight = constrain(pwmRight, -MAX_PWM, MAX_PWM);

  driveMotorLeft(pwmLeft);
  driveMotorRight(pwmRight);
}

void driveMotorLeft(int pwm) {
  if (abs(pwm) < 10) {
    // Stop Left Motor
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, LOW);
    analogWrite(ENA_PIN, 0);
  } else if (pwm > 0) {
    // Forward Left Motor
    digitalWrite(IN1_PIN, HIGH);
    digitalWrite(IN2_PIN, LOW);
    int activePwm = map(pwm, 1, 255, MIN_PWM, 255);
    analogWrite(ENA_PIN, activePwm);
  } else {
    // Reverse Left Motor
    digitalWrite(IN1_PIN, LOW);
    digitalWrite(IN2_PIN, HIGH);
    int activePwm = map(abs(pwm), 1, 255, MIN_PWM, 255);
    analogWrite(ENA_PIN, activePwm);
  }
}

void driveMotorRight(int pwm) {
  if (abs(pwm) < 10) {
    // Stop Right Motor
    digitalWrite(IN3_PIN, LOW);
    digitalWrite(IN4_PIN, LOW);
    analogWrite(ENB_PIN, 0);
  } else if (pwm > 0) {
    // Forward Right Motor
    digitalWrite(IN3_PIN, HIGH);
    digitalWrite(IN4_PIN, LOW);
    int activePwm = map(pwm, 1, 255, MIN_PWM, 255);
    analogWrite(ENB_PIN, activePwm);
  } else {
    // Reverse Right Motor
    digitalWrite(IN3_PIN, LOW);
    digitalWrite(IN4_PIN, HIGH);
    int activePwm = map(abs(pwm), 1, 255, MIN_PWM, 255);
    analogWrite(ENB_PIN, activePwm);
  }
}

void stopMotors() {
  currentLinearV = 0.0;
  currentAngularW = 0.0;
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  digitalWrite(IN3_PIN, LOW);
  digitalWrite(IN4_PIN, LOW);
  analogWrite(ENA_PIN, 0);
  analogWrite(ENB_PIN, 0);
}

float readUltrasonicDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH, 15000); // 15ms timeout (~2.5m)
  if (duration == 0) return -1.0;
  return (duration * 0.0343) / 2.0;
}
