# 💊 Alzheimer Patient Pill Reminder System

An intelligent HCI pill reminder system that uses computer vision, gesture recognition, and ArUco markers to help Alzheimer patients take their medication on time. The system combines a Python-based tracking backend with a Java GUI frontend.

## 📋 Features

- **Visual Pill Box Detection**: Uses ArUco markers to detect when the pill box is in view
- **Hand Gesture Recognition**: Recognizes when the user picks up pills using the $1 gesture recognizer
- **Smart Reminders**: Time-based reminders with configurable dose schedules
- **Duplicate Detection**: Warns users if they try to take pills too soon after the last dose
- **Real-time Feedback**: Visual and audio notifications through a Java GUI
- **Activity Logging**: All interactions are logged to CSV for tracking compliance

## 🛠️ Technologies Used

### Python Backend (`PillTracker.py`)
- **OpenCV**: For camera input and ArUco marker detection
- **MediaPipe**: For hand tracking and landmark detection
- **dollarpy**: For gesture recognition using the $1 recognizer
- **pyttsx3**: For text-to-speech notifications
- **Socket Communication**: For real-time GUI updates

### Java Frontend (`PillReminderGUI`)
- **Swing**: For GUI components
- **Socket Client**: For receiving updates from Python backend
- **Audio Playback**: For alert sounds

## 📦 Installation

### Prerequisites
- Python 3.7+
- Java JDK 8+
- Webcam

### Python Dependencies
```bash
pip install opencv-python opencv-contrib-python
pip install mediapipe
pip install numpy
pip install pyttsx3
pip install dollarpy
```

### Java Setup
The Java GUI is set up as a NetBeans project. You can:
1. Open the `PillReminderGUI` folder in NetBeans
2. Build and run the project

## 🚀 Usage

### 1. Prepare Your Pill Box
- Print an ArUco marker (DICT_4X4_50, ID 0)
- Attach it to your pill box

### 2. Start the Python Tracker
```bash
python PillTracker.py
```

### 3. Run the Java GUI in NetBeans

### 4. Taking Pills
1. When it's time for your dose, the system will remind you
2. Hold your pill box in front of the camera
3. Reach toward the box to pick up pills
4. The system will detect your gesture and confirm the dose

## ⚙️ Configuration

Edit `PillTracker.py` to customize:

```python
# Dose Times (24-hour format)
DOSE_TIMES = {
    "morning": {"hour": 16, "minute": 0},  # 4:00 PM
    "night": {"hour": 23, "minute": 30}    # 11:30 PM
}

# Time Windows
REMINDER_WINDOW_MINS = 5   # Show reminder 5 mins before
PILL_WINDOW_MINS = 30      # Stop reminder 30 mins after
WARNING_COOLDOWN_MINS = 60 # Warn if taken again within 60 mins

## 🎯 Gesture Recognition

The system recognizes the following gestures as "pill taking":
1. **L-shape**: Horizontal movement followed by downward movement
2. **Reverse L-shape**: Opposite direction
3. **Horizontal swipe**: Left-to-right or right-to-left
4. **Simple reach**: Moving hand toward the marker

**Tips for best recognition:**
- Make smooth, deliberate gestures
- Keep your hand visible to the camera
- Ensure good lighting
- Position the ArUco marker clearly in view

## 📊 Activity Logging

All events are logged to `pill_log.csv` with:
- Timestamp
- Action type (reminder, take_attempt, etc.)
- Pill box detection status
- Gesture classification
- Result status (SUCCESS, WARNING, IGNORED)

## 🎓 Academic

This project was developed for an HCI (Human-Computer Interaction) midterm. See the included documentation files for:
- PACT Analysis (People, Activities, Contexts, Technologies)
- Design scenarios and user specifications