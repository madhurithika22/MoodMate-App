import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Mock Emotion Detection Class
class MockEmotionModel:
    def predict(self, frame):
        # Simulating a prediction with random emotions
        emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
        return np.random.choice(emotions)

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = MockEmotionModel()

    def recv(self, frame):
        # Simulate emotion prediction on the frame
        emotion = self.model.predict(frame)
        st.session_state['emotion'] = emotion  # Save predicted emotion in session state
        return frame

# Streamlit App Layout
st.title("MoodMate: Webcam-Based Mood Detection")

# Show video streaming
webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_mode=True,
)

# Display the detected emotion
if 'emotion' in st.session_state:
    st.write(f"Detected Emotion: {st.session_state['emotion']}")

# Footer/Disclaimer
st.markdown("With love ❤️")
