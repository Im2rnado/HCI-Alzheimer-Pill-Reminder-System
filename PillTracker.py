import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np
import socket
import json
import threading
import time
from datetime import datetime, timedelta
import csv
import os
import pyttsx3
from dollarpy import Recognizer, Template, Point

# --- Configuration ---
# Socket Config
HOST = 'localhost'
PORT = 65432

# Time Config (Using 24-hour format)
# Using more accessible times for testing, e.g., 4:00 PM = 16:00
DOSE_TIMES = {
    "morning": {"hour": 16, "minute": 0},  # 4:00 PM
    "night": {"hour": 23, "minute": 30}    # 8:00 PM
}
REMINDER_WINDOW_MINS = 5  # Show reminder 5 mins before
PILL_WINDOW_MINS = 30     # Stop reminder 30 mins after
WARNING_COOLDOWN_MINS = 60 # Warn if taken again within 60 mins

# ArUco Marker Config
MARKER_ID = 0  # The ID of the marker on the pillbox
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters()

# Gesture Config
PROXIMITY_THRESHOLD = 80  # Pixels to trigger gesture recording
gesture_points = []
is_recording = False

# Template 1: "L-shape" (left-to-right, then down in screen coords)
template_1_points = [
    Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0), Point(40, 0), Point(50, 0),
    Point(50, 10), Point(50, 20), Point(50, 30), Point(50, 40), Point(50, 50)
]
template_take_pill_1 = Template('take_pill', template_1_points)
# Template 2: "Reverse L-shape" (right-to-left, then down in screen coords)
template_2_points = [
    Point(50, 0), Point(40, 0), Point(30, 0), Point(20, 0), Point(10, 0), Point(0, 0),
    Point(0, 10), Point(0, 20), Point(0, 30), Point(0, 40), Point(0, 50)
]
template_take_pill_2 = Template('take_pill', template_2_points)
# Template 3: Simple horizontal line (left-to-right)
template_3_points = [
    Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0), Point(40, 0), 
    Point(50, 0), Point(60, 0), Point(70, 0), Point(80, 0), Point(90, 0), Point(100, 0)
]
template_take_pill_3 = Template('take_pill', template_3_points)
# Template 4: Simple horizontal line (right-to-left)
template_4_points = [
    Point(100, 0), Point(90, 0), Point(80, 0), Point(70, 0), Point(60, 0),
    Point(50, 0), Point(40, 0), Point(30, 0), Point(20, 0), Point(10, 0), Point(0, 0)
]
template_take_pill_4 = Template('take_pill', template_4_points)
dollar_recognizer = Recognizer([template_take_pill_1, template_take_pill_2, template_take_pill_3, template_take_pill_4])

# MediaPipe Config
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# State Management
STATE_WAITING = "WAITING"
STATE_REMINDER = "REMINDER"
STATE_TAKEN = "TAKEN"
current_state = STATE_WAITING
current_dose_slot = None  # "morning" or "night"
last_pill_time = {"morning": None, "night": None}

# Logging
LOG_FILE = 'pill_log.csv'

# Voice Engine
try:
    voice_engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: Could not initialize voice engine: {e}")
    voice_engine = None

# --- Helper Functions ---

def speak(text):
    """Speaks the given text in a separate thread."""
    if not voice_engine:
        return
    try:
        def run_speak():
            voice_engine.say(text)
            voice_engine.runAndWait()
        threading.Thread(target=run_speak, daemon=True).start()
    except Exception as e:
        print(f"Error in speech thread: {e}")

def log_to_csv(action, pill_box_detected, gesture_class, status):
    """Appends a new event to the CSV log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'action', 'pill_box_detected', 'gesture_class', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write header if file is new
            
        writer.writerow({
            'timestamp': timestamp,
            'action': action,
            'pill_box_detected': pill_box_detected,
            'gesture_class': gesture_class,
            'status': status
        })

def send_to_gui(conn, event_type, message, data=None):
    """Sends a formatted JSON message to the Java GUI."""
    if conn:
        try:
            payload = {"event": event_type, "message": message, "data": data or {}}
            conn.sendall((json.dumps(payload) + "\n").encode('utf-8'))
        except (socket.error, BrokenPipeError) as e:
            print(f"Socket error: {e}. GUI disconnected.")
            conn = None  # Mark connection as dead
    return conn

def check_time_state():
    """Updates the global state based on the current time and last pill taken."""
    global current_state, current_dose_slot
    now = datetime.now()
    
    new_state = STATE_WAITING
    new_dose_slot = None
    
    for slot, dose_time in DOSE_TIMES.items():
        dose_dt = now.replace(hour=dose_time["hour"], minute=dose_time["minute"], second=0, microsecond=0)
        start_reminder = dose_dt - timedelta(minutes=REMINDER_WINDOW_MINS)
        end_reminder = dose_dt + timedelta(minutes=PILL_WINDOW_MINS)
        
        # Check if we are in a pill window
        if start_reminder <= now <= end_reminder:
            new_dose_slot = slot
            last_take = last_pill_time.get(slot)
            
            if last_take is None:
                new_state = STATE_REMINDER
            else:
                # Pill already taken. Check if we're in the warning cooldown.
                cooldown_end = last_take + timedelta(minutes=WARNING_COOLDOWN_MINS)
                if now <= cooldown_end:
                    new_state = STATE_TAKEN # "Taken" state implies cooldown
                else:
                    new_state = STATE_WAITING # Cooldown over, back to waiting
            break # Found our window, stop checking
            
    # State has changed
    if new_state != current_state and new_dose_slot != current_dose_slot:
        current_state = new_state
        current_dose_slot = new_dose_slot
        
        if current_state == STATE_REMINDER:
            msg = f"Time for your {current_dose_slot} pill ({dose_time['hour']}:{dose_time['minute']:02d})"
            speak(f"Please take your {current_dose_slot} pill.")
            log_to_csv("reminder_started", False, "N/A", msg)
            return "REMINDER", msg
        
        elif current_state == STATE_WAITING:
            return "STATUS", "All good. Waiting for next dose."
            
    # If no window, we are waiting
    if new_dose_slot is None:
        current_state = STATE_WAITING
        current_dose_slot = None
        
    return None, None # No change


def handle_pill_take_event(gesture_name):
    """Handles the logic when a 'take_pill' gesture is detected."""
    global current_state, last_pill_time, current_dose_slot
    
    now = datetime.now()
    action = "take_attempt"
    status = "N/A"
    
    # Check if this is a "take_pill" gesture
    if gesture_name != 'take_pill':
        log_to_csv(action, True, gesture_name, "wrong_gesture")
        return None, None # Not the gesture we care about

    # --- It IS a "take_pill" gesture ---
    
    # Case 1: We are in the reminder window and pill hasn't been taken
    if current_state == STATE_REMINDER and current_dose_slot:
        status = "SUCCESS"
        last_pill_time[current_dose_slot] = now
        current_state = STATE_TAKEN # Move to "taken" state (in cooldown)
        msg = f"{current_dose_slot.capitalize()} pill taken successfully!"
        
        log_to_csv(action, True, gesture_name, status)
        speak("Pill taken successfully.")
        return "SUCCESS", msg
        
    # Case 2: We are NOT in a reminder window, but check all recent doses
    else:
        for slot, last_take in last_pill_time.items():
            if last_take:
                cooldown_end = last_take + timedelta(minutes=WARNING_COOLDOWN_MINS)
                if now <= cooldown_end:
                    # This is a warning!
                    status = "WARNING"
                    msg = f"Warning: You already took the {slot} dose recently!"
                    log_to_csv(action, True, gesture_name, status)
                    speak("Warning: You already took this dose.")
                    return "WARNING", msg

    # Case 3: No recent pill, no reminder. Just a random gesture.
    status = "IGNORED"
    log_to_csv(action, True, gesture_name, status)
    return "STATUS", "Pillbox interaction detected, but it is not pill time."


# --- Main Function ---

def main():
    global is_recording, gesture_points
    
    # Setup Socket Server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"Server listening on {HOST}:{PORT}")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    
    conn = send_to_gui(conn, "STATUS", "System connected. Initializing...", None)
    
    # Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    last_state_check = time.time()
    last_gui_message = ""
    last_known_marker_center = None
    
    while cap.isOpened() and conn:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # --- 1. Time-based State Check (run once per second) ---
        if time.time() - last_state_check > 1.0:
            last_state_check = time.time()
            event, msg = check_time_state()
            if event and msg != last_gui_message:
                conn = send_to_gui(conn, event, msg)
                last_gui_message = msg

        # --- 2. ArUco Marker Detection ---
        corners, ids, _ = aruco.detectMarkers(frame, ARUCO_DICT, parameters=ARUCO_PARAMS)
        pillbox_detected = False
        
        if ids is not None and MARKER_ID in ids:
            pillbox_detected = True
            # Find the index of our marker
            idx = np.where(ids == MARKER_ID)[0][0]
            marker_corners = corners[idx][0]
            
            # Calculate center
            marker_center = np.mean(marker_corners, axis=0).astype(int)
            last_known_marker_center = marker_center
            
            cv2.circle(frame, tuple(marker_center), 10, (0, 255, 0), -1)
            
            # Draw the bounding box
            aruco.drawDetectedMarkers(frame, [corners[idx]])

        # --- 3. MediaPipe Hand Detection ---
        hand_results = hands.process(frame_rgb)
        hand_landmark = None
        
        if hand_results.multi_hand_landmarks:
            # Get landmarks for the first hand
            hand_lm = hand_results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip (Landmark 8)
            index_tip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand_landmark = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.circle(frame, hand_landmark, 10, (255, 0, 0), -1)
            

        # --- 4. Gesture Recognition Logic ---
        if hand_landmark and last_known_marker_center is not None:
            # Calculate distance
            dist = np.linalg.norm(np.array(last_known_marker_center) - np.array(hand_landmark))
            cv2.line(frame, tuple(last_known_marker_center), hand_landmark, (255, 255, 0), 2)
            
            # State: Hand is NEAR pillbox
            if dist < PROXIMITY_THRESHOLD and not is_recording:
                print("Started recording gesture...")
                is_recording = True
                gesture_points = [] # Clear previous points
                log_to_csv("gesture_start", pillbox_detected, "N/A", "Hand near marker")

            # State: Hand is being tracked
            if is_recording:
                gesture_points.append(Point(hand_landmark[0], hand_landmark[1]))
                cv2.putText(frame, "RECORDING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # State: Hand has moved AWAY
                if dist > (PROXIMITY_THRESHOLD + 20): # Add hysteresis
                    print(f"Stopped recording. Points: {len(gesture_points)}")
                    is_recording = False
                    
                    if len(gesture_points) > 10: # Need a minimum number of points    
                        # Recognize the gesture
                        result = dollar_recognizer.recognize(gesture_points)
                        gesture_name, score = result
                        score = round(score, 2)
                        
                        # Debug: Show gesture path info
                        first_pt = gesture_points[0]
                        last_pt = gesture_points[-1]
                        dx = last_pt.x - first_pt.x
                        dy = last_pt.y - first_pt.y
                        print(f"Gesture: {gesture_name}, Score: {score}, Points: {len(gesture_points)}, Delta: ({dx:.0f}, {dy:.0f})")
                        
                        if score > 0.25: # Confidence threshold (lowered from 0.3)
                            # --- This is the main event! ---
                            event, msg = handle_pill_take_event(gesture_name)
                            if event:
                                conn = send_to_gui(conn, event, msg)
                                last_gui_message = msg
                        else:
                            log_to_csv("take_attempt", True, "unknown", f"low_score_gesture ({score})")
                            
                    gesture_points = [] # Clear for next time
                    
        elif not hand_landmark and is_recording:
            print("Hand lost, cancelling gesture.")
            is_recording = False
            gesture_points = []

        # --- 5. Display Video ---
        cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Pill Tracker', frame)

        if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to quit
            break

    # --- Cleanup ---
    print("Shutting down...")
    if conn:
        conn.close()
    server_socket.close()
    cap.release()
    cv2.destroyAllWindows()
    log_to_csv("system_shutdown", False, "N/A", "System stopped")

if __name__ == "__main__":
    # Initialize log file
    log_to_csv("system_startup", False, "N/A", "System started")
    main()