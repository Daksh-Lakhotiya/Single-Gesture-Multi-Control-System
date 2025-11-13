import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import os
import tensorflow as tf
import subprocess
import psutil
import serial
import serial.tools.list_ports

# ====== Load trained gesture model ======
MODEL_PATH = 'gesture_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("gesture_model.h5 not found in project folder.")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Gesture model loaded successfully")

# ====== Arduino setup ======
ARDUINO_BAUD = 9600
arduino = None

def connect_arduino():
    global arduino
    port = 'COM7'  # update if needed
    try:
        arduino = serial.Serial(port, ARDUINO_BAUD, timeout=1)
        time.sleep(2)
        print(f"✅ Arduino connected on {port}")
    except Exception as e:
        print(f"⚠️ Could not connect to Arduino: {e}")
        arduino = None

def send_to_arduino(cmd):
    if arduino and arduino.is_open:
        try:
            arduino.write(f"{cmd}\n".encode())
            arduino.flush()
            print(f"➡️ Sent to Arduino: {cmd}")
        except Exception as e:
            print(f"⚠️ Error sending to Arduino: {e}")
    else:
        print(f"⚠️ Arduino not connected. Command '{cmd}' skipped.")

connect_arduino()

# ====== Buffers and constants ======
landmarks_buffer = deque(maxlen=30)
COOLDOWN_TIME = 2.0
last_gesture_time = 0
confidence_threshold = 0.8

# ====== Mediapipe setup ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5,
                                  refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# ====== States ======
current_mode = "PC"
mode_names = {"PC": "PC (Gallery Swipe)", "light": "Light Control", "fan": "Fan Control"}
light_status = False
fan_status = False
gallery_open = False

# ====== PC MODE (IrfanView) ======
irfanview_path = r"C:\Program Files\IrfanView\i_view64.exe"
screenshots_folder = r"C:\Users\njosh\OneDrive\Pictures\Screenshots"
images = sorted([f for f in os.listdir(screenshots_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
current_image_index = 0
current_process = None

def ensure_irfan_closed():
    for proc in psutil.process_iter(['name']):
        try:
            if 'i_view64.exe' in proc.info['name'].lower():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(0.5)

def open_gallery():
    global gallery_open, current_image_index, current_process
    if gallery_open or not images:
        return
    ensure_irfan_closed()
    image_path = os.path.join(screenshots_folder, images[current_image_index])
    current_process = subprocess.Popen([irfanview_path, image_path])
    gallery_open = True
    print(f"🖼️ Gallery opened: {images[current_image_index]}")

def close_gallery():
    global gallery_open, current_process
    if gallery_open:
        ensure_irfan_closed()
        gallery_open = False
        current_process = None
        print("❌ Gallery closed")

def next_image():
    global current_image_index, current_process
    if not gallery_open:
        return
    current_image_index = (current_image_index + 1) % len(images)
    ensure_irfan_closed()
    image_path = os.path.join(screenshots_folder, images[current_image_index])
    current_process = subprocess.Popen([irfanview_path, image_path])
    print(f"➡️ Next image: {images[current_image_index]}")

# ====== GAZE DETECTION ======
gaze_history = deque(maxlen=5)
def smoothed_gaze(avg_ratio):
    gaze_history.append(avg_ratio)
    return np.mean(gaze_history)

def get_gaze_direction(face_landmarks, w, h):
    left_eye_inner, left_eye_outer = 33, 133
    right_eye_inner, right_eye_outer = 362, 263
    left_iris_center, right_iris_center = 468, 473

    def coord(i):
        lm = face_landmarks.landmark[i]
        return np.array([lm.x * w, lm.y * h])

    left_ratio = (coord(left_iris_center)[0] - coord(left_eye_inner)[0]) / \
                 (coord(left_eye_outer)[0] - coord(left_eye_inner)[0])
    right_ratio = (coord(right_iris_center)[0] - coord(right_eye_inner)[0]) / \
                  (coord(right_eye_outer)[0] - coord(right_eye_inner)[0])
    avg_ratio = smoothed_gaze((left_ratio + right_ratio) / 2)

    if avg_ratio < 0.42:
        return "left"
    elif avg_ratio > 0.58:
        return "right"
    else:
        return "center"

# ====== RELAY CONTROL ======
def toggle_light(on):
    global light_status
    if on:
        if not light_status:
            send_to_arduino("LIGHT_ON")
            light_status = True
            print("💡 Light turned ON")
        else:
            print("💡 Light already ON")
    else:
        if light_status:
            send_to_arduino("LIGHT_OFF")
            light_status = False
            print("💡 Light turned OFF")
        else:
            print("💡 Light already OFF")

def toggle_fan(on):
    global fan_status
    if on:
        if not fan_status:
            send_to_arduino("FAN_ON")
            fan_status = True
            print("🌀 Fan turned ON")
        else:
            print("🌀 Fan already ON")
    else:
        if fan_status:
            send_to_arduino("FAN_OFF")
            fan_status = False
            print("🌀 Fan turned OFF")
        else:
            print("🌀 Fan already OFF")

# ====== MAIN LOOP ======
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hands_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # Eye-based mode switch
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            gaze = get_gaze_direction(face_landmarks, w, h)
            if gaze == "left":
                current_mode = "light"
            elif gaze == "right":
                current_mode = "fan"
            else:
                current_mode = "PC"

    # Gesture Detection
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            landmarks_buffer.append(coords)

            if len(landmarks_buffer) == 30:
                input_data = np.expand_dims(np.array(landmarks_buffer), axis=0)
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                if confidence >= confidence_threshold:
                    gesture = "left_swipe" if predicted_class == 0 else "right_swipe"
                    current_time = time.time()
                    if (current_time - last_gesture_time) > COOLDOWN_TIME:
                        last_gesture_time = current_time
                        print(f"🖐 Gesture: {gesture} in mode: {current_mode}")

                        if current_mode == "PC":
                            if gesture == "left_swipe":
                                if not gallery_open:
                                    open_gallery()
                                else:
                                    next_image()
                            elif gesture == "right_swipe":
                                close_gallery()
                        elif current_mode == "light":
                            if gesture == "left_swipe":
                                toggle_light(True)
                            elif gesture == "right_swipe":
                                toggle_light(False)
                        elif current_mode == "fan":
                            if gesture == "left_swipe":
                                toggle_fan(True)
                            elif gesture == "right_swipe":
                                toggle_fan(False)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        landmarks_buffer.clear()

    # === Display info on window ===
    cv2.putText(image, f"MODE: {mode_names[current_mode]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    if current_mode == "light":
        color = (0, 255, 0) if light_status else (0, 0, 255)
        cv2.putText(image, f"Light: {'ON' if light_status else 'OFF'}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    elif current_mode == "fan":
        color = (0, 255, 0) if fan_status else (0, 0, 255)
        cv2.putText(image, f"Fan: {'ON' if fan_status else 'OFF'}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    elif current_mode == "PC":
        status = "Gallery OPEN" if gallery_open else "Gallery CLOSED"
        cv2.putText(image, f"PC: {status}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gesture and Eye Control", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
if arduino and arduino.is_open:
    arduino.close()
    print("🔌 Arduino connection closed.")