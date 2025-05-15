# ✅ AsanaAI - FINAL Webcam Live Pose Detection Script (Fixed)
import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import joblib

# Load trained model
model = joblib.load("pose_classifier_26April.pkl")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# Initialize audio
pygame.mixer.init()
correct_audio = "correct.mp3"
incorrect_audio = "alert.mp3"

# Open webcam
cap = cv2.VideoCapture(0)

# Cooldown timers
last_alert_time = 0
COOLDOWN = 5  # seconds to wait before playing audio again

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    prediction = "Unknown"

    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

        if len(keypoints) == 132:
            keypoints = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints)[0]
            
            # Draw pose
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Based on prediction or pose detection
    now = time.time()
    if prediction == "Unknown" or "incorrect" in prediction.lower() or "wrong" in prediction.lower():
        label = "⚠️ Incorrect Pose Detected"
        color = (0, 0, 255)  # Red
        if now - last_alert_time > COOLDOWN:
            pygame.mixer.music.load(incorrect_audio)
            pygame.mixer.music.play()
            last_alert_time = now
    else:
        label = f"✅ {prediction}"
        color = (0, 255, 0)  # Green
        if now - last_alert_time > COOLDOWN:
            pygame.mixer.music.load(correct_audio)
            pygame.mixer.music.play()
            last_alert_time = now

    # Display label
    cv2.putText(frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('AsanaAI Webcam Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
