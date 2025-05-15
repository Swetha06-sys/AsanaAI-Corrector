# Webcam Real-time Pose Detection + Audio Alerts
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from gtts import gTTS
import os

# Load trained model
clf = joblib.load('pose_classifier_26April.pkl')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Audio play function
def play_audio(message, filename="temp_audio.mp3"):
    tts = gTTS(text=message, lang='en')
    tts.save(filename)
    os.system(f"start {filename}")  # Windows
    # os.system(f"afplay {filename}")  # Mac

# Function to extract keypoints from frame
def extract_keypoints_from_frame(image):
    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if result.pose_landmarks:
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(keypoints).reshape(1, -1), result
    return None, None

# Setup webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# Setup alert cooldown
last_alert_time = 0
cooldown_seconds = 5  # play alert every 5 seconds only

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame

    keypoints, result = extract_keypoints_from_frame(frame)

    if keypoints is not None:
        prediction = clf.predict(keypoints)[0]

        now = time.time()

        if "incorrect" in prediction.lower():
            color = (0, 0, 255)  # Red for incorrect
            label = f"⚠️ Incorrect: Adjust!"
            if now - last_alert_time > cooldown_seconds:
                play_audio("Incorrect pose detected. Please adjust your posture.")
                last_alert_time = now
        else:
            color = (0, 255, 0)  # Green for correct
            label = f"✅ Correct Pose: {prediction}"
            if now - last_alert_time > cooldown_seconds:
                play_audio("Good job! Correct pose detected.")
                last_alert_time = now

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Display label
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    else:
        cv2.putText(frame, "⚠️ Pose Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('AsanaAI - Webcam Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
