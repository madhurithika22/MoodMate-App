import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace
from PIL import Image
import numpy as np
import av

st.set_page_config(page_title="MoodMate - Real-time Mood Detector", layout="centered")
st.title("😊 MoodMate - Real-time Mood Detector")
st.markdown("Detect your **mood** instantly through webcam!")

mood_placeholder = st.empty()

# Define emotion detection logic
def detect_emotion(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        return "Error"

# Streamlit WebRTC video transformer
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = img[:, :, ::-1]  # Convert BGR to RGB
        emotion = detect_emotion(Image.fromarray(img_rgb))

        # Display mood
        mood_placeholder.markdown(f"### 🧠 Detected Mood: **{emotion.capitalize()}**")
        return frame

# Start webcam stream
webrtc_streamer(
    key="mood_detector",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

st.markdown("---")
st.info("Built with ❤️")