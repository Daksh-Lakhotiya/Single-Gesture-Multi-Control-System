# Single Gesture Multi Control System

This project implements a real-time gesture and eye-gaze-based control system that enables users to control PC applications and IoT devices (like lights and fans) using a single trained gesture model.  
It combines MediaPipe, TensorFlow, OpenCV, and Arduino for seamless multi-mode operation.

---

## Features

- **Eye Gaze-Based Mode Switching**
  - Look Left → Light Control Mode  
  - Look Right → Fan Control Mode  
  - Look Center → PC Gallery Mode  

- **Gesture-Based Actions**
  - Left Swipe → Turn ON / Open / Next  
  - Right Swipe → Turn OFF / Close  

- **Supported Modes**
  1. Light Control — Toggle ON/OFF using gestures  
  2. Fan Control — Toggle ON/OFF using gestures  
  3. PC Mode — Open and navigate image gallery  

- **Hardware Integration**
  - Works with Arduino via serial communication  
  - Controls external devices through relays  

- **Machine Learning**
  - Uses a CNN model (`gesture_model.h5`) for gesture recognition  

---

## System Architecture

| Component | Description |
|------------|-------------|
| Camera | Captures real-time video feed |
| MediaPipe Hands & FaceMesh | Detects hand and eye landmarks |
| TensorFlow Model | Classifies gestures (left/right swipe) |
| Arduino UNO | Executes control commands for connected devices |
| PC (Windows) | Controls image viewer (IrfanView) based on gestures |

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Single-Gesture-Multi-Control-System.git
cd Single-Gesture-Multi-Control-System
```

### 2. Install Dependencies
```bash
pip install opencv-python mediapipe numpy tensorflow psutil pyserial
```

### 3. Place Model File
Make sure `gesture_model.h5` (the trained gesture recognition model) is located in the project directory.

---

## Hardware Setup

**Required Components:**
- Arduino UNO  
- 2-Channel Relay Module  
- Fan and Light (as load devices)  
- USB Cable  

**Connections:**

| Arduino Pin | Relay Channel | Device |
|--------------|----------------|--------|
| D8 | IN1 | Light |
| D9 | IN2 | Fan |
| 5V, GND | VCC, GND | Relay Power |

Update the COM port in the Python script if required:
```python
port = 'COM7'  # Change this according to your system
```

---

## How It Works

1. The camera captures a live video feed.
2. MediaPipe detects hand landmarks and eye gaze direction.
3. Eye gaze determines the active mode:
   - Left → Light Control
   - Right → Fan Control
   - Center → PC Mode
4. The TensorFlow model predicts gestures (left/right swipe).
5. Based on the mode and gesture, corresponding actions are executed:
   - Light/Fan mode: Relay control through Arduino.
   - PC mode: Image gallery control via IrfanView.

---

## Commands Sent to Arduino

| Gesture | Mode | Command Sent | Action |
|----------|------|---------------|--------|
| Left Swipe | Light | LIGHT_ON | Turn Light ON |
| Right Swipe | Light | LIGHT_OFF | Turn Light OFF |
| Left Swipe | Fan | FAN_ON | Turn Fan ON |
| Right Swipe | Fan | FAN_OFF | Turn Fan OFF |

---

## PC Mode Details

- Uses **IrfanView** to display images from the screenshots folder:
  ```python
  screenshots_folder = r"C:\Users\<username>\OneDrive\Pictures\Screenshots"
  ```
- Left Swipe → Next Image  
- Right Swipe → Close Gallery  
- You can change the folder path according to your system.

---

## Model Information

- Model File: `gesture_model.h5`
- Input Shape: `(1, 30, 63)` → 30 frames × 21 landmarks × (x, y, z)
- Output Classes:
  - 0 → Left Swipe  
  - 1 → Right Swipe
- Confidence Threshold: 0.8

---

## How to Run

1. Connect the Arduino to your PC.  
2. Ensure the correct COM port is set in the code.  
3. Run the main Python file:
   ```bash
   python main.py
   ```
4. Use your eyes to switch modes and gestures to perform actions.  
5. Press **ESC** to exit safely.

---

## Future Improvements

- Add more gesture types (e.g., up/down swipe).  
- Integrate voice commands.  
- Enable IoT-based remote control via Wi-Fi.  
- Add support for multiple devices and users.

---

## Author

**Daksh Lakhotiya**  
B.Tech Computer Science, VIT
