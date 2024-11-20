import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import io
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import json
from streamlit_option_menu import option_menu

# ========== Helper Functions ==========

# Function to extract MFCC for Fake Audio Detection
def extract_mfcc_fake(audio_path, n_mfcc=40, max_pad_len=174):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.flatten()
    except Exception as e:
        return None


# Function to extract MFCC for Speaker Identification
def extract_mfcc_speaker(uploaded_file):
    try:
        audio_data = uploaded_file.getvalue()
        y, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if mfcc.shape[1] == 0:  # Check if MFCC has time axis
            return None, None, None
        return np.mean(mfcc.T, axis=0), y, sr
    except Exception as e:
        return None, None, None


# Function to extract MFCC for Emotion Detection
def extract_mfcc_emotion(filepath, n_mfcc=40):
    audio, sr = librosa.load(filepath, sr=None, res_type="kaiser_fast")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


# Function to load Lottie animation
def load_lottiefile(filepath: str):
    with open(filepath) as f:
        return json.load(f)

# ========== Streamlit Dashboard ==========

# Set up the Streamlit page configuration

st.set_page_config(
    page_title="Dashboard",
    page_icon="assets/favicon.png",
    layout="wide",
)

# Layout for the app header
col1, col2 = st.columns([4, 3]) 
with col1:
    st.title("")
    # st.title("")
    st.title("AudioXplore Dashboard")
    st.markdown("Upload an audio file to get all the detailed Audio Analysis results!")

# with col2:
#     loader_main = load_lottiefile("assets/loader.json")  # Path to your loader.json file
#     st_lottie(
#         loader_main,
#         speed=1,
#         reverse=False,
#         loop=True,
#         quality="high", 
#         height=300,
#         width=600,
#         key=None,
#     )

# ========== Mode Selector (Main Page) ==========

# Real-time and Upload selector at the top of the main content
selected = option_menu(
    None, 
    options=["Upload", "Realtime"],  # Added "Realtime" option
    icons=["soundwave", "mic"],  # Icons for "Upload" and "Realtime"
    menu_icon="cast",
    default_index=0,  # Default to "Upload"
    orientation="horizontal"
)

# st.subheader("Upload Audio File" if selected == "Upload" else "Real-time Audio Analysis")

# Add a sidebar for quick navigation
with st.sidebar:
    st.header("Navigation")
    st.markdown("Choose an audio file to analyze.")
    # st.image("assets/fake_detect.png", use_container_width=True)  # Add a logo or image in the sidebar
    loader_main = load_lottiefile("assets/loader.json")  # Path to your loader.json file
    st_lottie(
        loader_main,
        speed=1,
        reverse=False,
        loop=True,
        quality="high", 
        height=300,
        width=300,
        key=None,
    )

# ========== Upload Mode Logic ==========

if selected == "Upload":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg", "aac", "m4a"])
    start_analysis = st.button("Start Analysis")

    if uploaded_file is not None:
        if start_analysis:
            with st.spinner("Analyzing the audio file..."):
                # Perform the analysis process

                # ========== Speaker Identification ========== 
                st.header("👤 Speaker Identification")
                try:
                    speaker_model = tf.keras.models.load_model("models/speaker_recognition_model.h5")
                    label_encoder = joblib.load("models/label_encoder.pkl")
                    mfcc_speaker, y, sr = extract_mfcc_speaker(uploaded_file)
                    if mfcc_speaker is not None:
                        mfcc_speaker = mfcc_speaker[np.newaxis, ..., np.newaxis]
                        pred_speaker = speaker_model.predict(mfcc_speaker)
                        speaker = label_encoder.inverse_transform([np.argmax(pred_speaker)])[0]
                        st.success(f"Predicted Speaker: **{speaker}**")
                    else:
                        st.error("Failed to process audio for Speaker Identification.")
                except Exception as e:
                    st.error(f"Error in Speaker Identification: {e}")

                # ========== Fake Audio Detection ========== 
                st.header("🕵️‍♂️ Fake Audio Detection")
                try:
                    fake_model = tf.keras.models.load_model("models/fake_audio_detection_model.h5")
                    scaler = joblib.load("models/scaler.pkl")
                    temp_path = os.path.join("temp_audio", uploaded_file.name)
                    os.makedirs("temp_audio", exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    mfcc_fake = extract_mfcc_fake(temp_path)
                    if mfcc_fake is not None:
                        scaled_features = scaler.transform([mfcc_fake])
                        pred_fake = fake_model.predict(scaled_features).flatten()
                        label_fake = "Real" if pred_fake < 0.5 else "Fake"
                        st.success(f"Audio Classification: **{label_fake}**")
                    else:
                        st.error("Failed to process audio for Fake Audio Detection.")
                except Exception as e:
                    st.error(f"Error in Fake Audio Detection: {e}")

                # ========== Speech Emotion Detection ========== 
                st.header("🎭 Speech Emotion Detection")
                try:
                    emotion_model = tf.keras.models.load_model("models/emotion_detection_model.h5")
                    emotion_dict = {
                        0: "Angry", 1: "Calm", 2: "Happy", 3: "Sad",
                        4: "Neutral", 5: "Fearful", 6: "Disgust",
                        7: "Surprised", 8: "Pleasant Surprise"
                    }
                    mfcc_emotion = extract_mfcc_emotion(temp_path)
                    if mfcc_emotion is not None:
                        mfcc_emotion = mfcc_emotion[np.newaxis, ..., np.newaxis]
                        pred_emotion = emotion_model.predict(mfcc_emotion)
                        emotion = emotion_dict[np.argmax(pred_emotion)]
                        st.success(f"Detected Emotion: **{emotion}**")
                    else:
                        st.error("Failed to process audio for Emotion Detection.")
                except Exception as e:
                    st.error(f"Error in Speech Emotion Detection: {e}")
        else:
            st.info("Please upload an audio file and click 'Start Analysis' to begin.")

    # Clean up temporary files
    if uploaded_file:
        temp_path = os.path.join("temp_audio", uploaded_file.name)
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ========== Real-time Mode Logic ==========

elif selected == "Realtime":
    st.info("Real-time audio analysis is coming soon! Stay tuned.")







