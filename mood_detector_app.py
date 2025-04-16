import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from fer import FER
import av
import numpy as np
from PIL import Image

st.set_page_config(page_title="MoodMate", layout="centered")
st.title("😊 MoodMate - AI Mood Detector")
st.write("Let’s detect your mood from your face using your webcam!")

# Load FER emotion detector
detector = FER(mtcnn=True)

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to BGR format

        # Detect emotions using FER (returns best detected face and emotions)
        result = detector.top_emotion(img)
        
        # Draw results on frame
        if result is not None:
            emotion, score = result
            text = f"{emotion} ({score*100:.1f}%)"
            cv_frame = img.copy()
            cv2.putText(cv_frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return cv_frame
        else:
            return img

# Stream from webcam
webrtc_streamer(
    key="mood",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.info("Make sure your webcam is turned on. Your mood will be detected live.")