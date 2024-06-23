
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
import mediapipe as mp
import webbrowser
import time
import os
import tensorflow as tf

# Load the TensorFlow Keras model
model_path = r"path-to-your-model.h5_file"
model = tf.keras.models.load_model(model_path)

# Mock emotion labels for demonstration purposes
label = np.array(['Angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'Surprise'])
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

# Initialize session state variables
if "run" not in st.session_state:
    st.session_state["run"] = True

try:
    emotion = np.load("emotion.npy")[0]
except Exception as e:
    emotion = ""
    st.write(f"Error loading emotion.npy: {e}")

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.holis = holistic.Holistic()
        self.start_time = time.time()
        self.emotion_detected = False

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = self.holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            
            if not self.emotion_detected and time.time() - self.start_time >= 5:
                # Use loaded model to predict emotion
                # Replace this with actual model prediction logic
                pred = np.random.choice(label)
                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
                self.emotion_detected = True
                st.session_state["run"] = False

        drawing.draw_landmarks(
            frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1)
        )
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Language")
singer = st.text_input("Singer")

if lang and singer and st.session_state["run"]:
    webrtc_streamer(
        key="key",
        video_processor_factory=EmotionProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
    )

if not st.session_state["run"]:
    webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+songs of+{singer}")
    np.save("emotion.npy", np.array([""]))
    st.session_state["run"] = True

# Accessing images from dataset directory
dataset_dir = r"path-to-your-trained_dataset"  # Update with your dataset directory path


if os.path.exists(dataset_dir):
    file_list = os.listdir(dataset_dir)
    for file_name in file_list:
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(dataset_dir, file_name)
            image = cv2.imread(file_path)
            if image is None:
                st.write(f"Error loading image: {file_path}")
            else:
                st.image(image, caption=file_name, use_column_width=True)
else:
    st.write(f"Dataset directory '{dataset_dir}' not found.")


