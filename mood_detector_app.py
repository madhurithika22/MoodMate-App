import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from deepface import DeepFace
import av
import cv2

# Streamlit app settings
st.set_page_config(page_title="MoodMate - Real-Time Mood Detector")
st.title("😊 MoodMate: Real-Time Mood Detector")
st.markdown("Let AI read your mood through your expressions in real-time!")

# Mood messages
feedback_map = {
    'happy': "Keep smiling! 😊",
    'sad': "Everything will be okay. You're strong. 💪",
    'angry': "Take a deep breath. Stay calm. 🧘",
    'surprise': "Wow! That’s unexpected! 😮",
    'fear': "Don't worry, you're safe. 💖",
    'disgust': "Yikes! Want to talk about it? 😬",
    'neutral': "Chilling like a pro. 😎"
}

# Custom video processor
class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.result_text = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        try:
            # Detect emotion
            result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]["dominant_emotion"]

            # Build text and choose message
            self.result_text = f"Mood: {dominant_emotion.capitalize()} | {feedback_map.get(dominant_emotion, '')}"

            # Draw text at the bottom of the frame
            height = image.shape[0]
            cv2.putText(image, self.result_text, (30, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error during emotion detection:", e)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Webcam stream interface
webrtc_streamer(
    key="mood-mate",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)