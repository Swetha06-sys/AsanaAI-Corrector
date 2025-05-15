import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time
from collections import defaultdict

# Load model and mediapipe
clf = joblib.load("pose_classifier_26April.pkl")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Session state setup
if "play_alert_audio" not in st.session_state:
    st.session_state.play_alert_audio = False
if "play_correct_audio" not in st.session_state:
    st.session_state.play_correct_audio = False
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "total_count" not in st.session_state:
    st.session_state.total_count = 0
if "webcam_correct" not in st.session_state:
    st.session_state.webcam_correct = 0
if "webcam_total" not in st.session_state:
    st.session_state.webcam_total = 0
if "webcam_pose_label" not in st.session_state:
    st.session_state.webcam_pose_label = ""
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0
if "pose_tracking" not in st.session_state:
    st.session_state.pose_tracking = defaultdict(lambda: {"correct": 0, "incorrect": 0})

st.set_page_config(page_title="AsanaAI - Yoga Pose Classifier", layout="centered")
st.title("🧘‍♀️ AsanaAI - Yoga Pose Classifier")
st.markdown("Upload a yoga pose image or use your webcam to check if it's **correct** or **incorrect**.")

def extract_keypoints_from_image(image):
    try:
        img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = pose.process(img_rgb)
    except Exception as e:
        st.error(f"Image conversion error: {e}")
        return None, None
    if result.pose_landmarks:
        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(keypoints).reshape(1, -1), result
    return None, result

st.subheader("📂 Upload a Yoga Pose Image")
uploaded_file = st.file_uploader("Choose a yoga pose image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Pose", use_column_width=True)
    keypoints, result = extract_keypoints_from_image(image)
    if keypoints is not None:
        prediction = clf.predict(keypoints)[0]
        st.success(f"✅ Predicted Pose: **{prediction}**")
        st.session_state.total_count += 1
        if "incorrect" not in prediction.lower() and "unknown" not in prediction.lower():
            st.session_state.correct_count += 1
            st.session_state.pose_tracking[prediction]["correct"] += 1
            st.session_state.play_correct_audio = True
        else:
            st.session_state.pose_tracking[prediction]["incorrect"] += 1
            st.session_state.play_alert_audio = True
        accuracy = (st.session_state.correct_count / st.session_state.total_count) * 100
        st.info(f"📊 Session Accuracy: **{accuracy:.2f}%**")
        annotated_image = np.array(image)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        st.image(annotated_image, caption="Pose Landmarks", channels="BGR", use_column_width=True)
    else:
        st.warning("⚠️ Pose not detected. Please upload a full-body image.")
        st.session_state.play_alert_audio = True

if st.button("🔄 Reset Performance"):
    st.session_state.correct_count = 0
    st.session_state.total_count = 0
    st.session_state.webcam_correct = 0
    st.session_state.webcam_total = 0
    st.session_state.pose_tracking = defaultdict(lambda: {"correct": 0, "incorrect": 0})

st.subheader("📷 Real-Time Webcam Pose Detection")

class PoseDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        prediction = "Unknown"

        if result.pose_landmarks:
            keypoints = []
            for lm in result.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

            if len(keypoints) == 132:
                prediction = clf.predict([keypoints])[0]
                st.session_state.webcam_pose_label = prediction
                st.session_state.webcam_total += 1
                now = time.time()

                if "incorrect" in prediction.lower() or "unknown" in prediction.lower():
                    st.session_state.pose_tracking[prediction]["incorrect"] += 1
                    if now - st.session_state.last_alert_time > 5:
                        st.session_state.play_alert_audio = True
                        st.session_state.last_alert_time = now
                else:
                    st.session_state.webcam_correct += 1
                    st.session_state.pose_tracking[prediction]["correct"] += 1
                    if now - st.session_state.last_alert_time > 5:
                        st.session_state.play_correct_audio = True
                        st.session_state.last_alert_time = now
        else:
            prediction = "Unknown"
            st.session_state.webcam_pose_label = prediction
            st.session_state.play_alert_audio = True

        label = (
            f"✅ {prediction}"
            if "incorrect" not in prediction.lower() and "unknown" not in prediction.lower()
            else "⚠️ Adjust your posture!"
        )
        color = (0, 255, 0) if "incorrect" not in prediction.lower() and "unknown" not in prediction.lower() else (0, 0, 255)

        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="asanaai-webcam",
    video_processor_factory=PoseDetector,
    media_stream_constraints={"video": True, "audio": False}
)

if st.session_state.webcam_pose_label:
    st.info(f"📸 Webcam Pose Prediction: **{st.session_state.webcam_pose_label}**")

if st.session_state.play_alert_audio:
    st.audio("alert.mp3", format="audio/mp3", start_time=0)
    st.session_state.play_alert_audio = False

if st.session_state.play_correct_audio:
    st.audio("correct.mp3", format="audio/mp3", start_time=0)
    st.session_state.play_correct_audio = False

st.subheader("📌 Pose-by-Pose Tracking")
pose_table = []
for pose_name, counts in st.session_state.pose_tracking.items():
    pose_table.append([pose_name, counts["correct"], counts["incorrect"]])
st.table(pd.DataFrame(pose_table, columns=["Pose", "Correct", "Incorrect"]))