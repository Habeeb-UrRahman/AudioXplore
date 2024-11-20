# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os
# from streamlit_option_menu import option_menu
# from streamlit_lottie import st_lottie
# import json

# # Load the LSTM model
# model = load_model("models\emotion_detection_model.h5")

# # Emotion mapping (same as during training)
# emotion_dict = {
#     0: 'Angry',
#     1: 'Calm',
#     2: 'Happy',
#     3: 'Sad',
#     4: 'Neutral',
#     5: 'Fearful',
#     6: 'Disgust',
#     7: 'Surprised',
#     8: 'Pleasant Surprise'
# }

# # Load Lottie animation
# def load_lottiefile(filepath: str):
#     with open(filepath) as f:
#         return json.load(f)

# # Configure Streamlit layout
# st.set_page_config(page_title="Voice Emotion Recognition", layout="wide")

# # Layout for the app header
# col1, col2 = st.columns(2)
# with col1:
#     st.title("Voice Brew")
#     st.subheader("Detect speech emotions in real-time!")

# with col2:
#     loader_main = load_lottiefile("assets\loader.json")  # Path to your loader.json file
#     st_lottie(
#         loader_main,
#         speed=1,
#         reverse=False,
#         loop=True,
#         quality="high", 
#         height=300,
#         width=800,
#         key=None,
#     )

# # Option menu for selecting between "Upload" and "Realtime"
# selected = option_menu(
#     None, 
#     options=["Upload", "Realtime"],  # Added "Realtime" option
#     icons=["soundwave", "mic"],  # Icons for "Upload" and "Realtime"
#     menu_icon="cast",
#     default_index=0,  # Default to "Upload"
#     orientation="horizontal"
# )

# # Function to extract MFCC features from an audio file
# def extract_features(filepath, n_mfcc=40):
#     audio, sr = librosa.load(filepath, sr=None, res_type="kaiser_fast")
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
#     mfccs = np.mean(mfccs.T, axis=0)
#     return mfccs

# # Function to plot waveform and spectrogram
# def plot_waveform_and_spectrogram(audio, sr):
#     fig, ax = plt.subplots(2, 1, figsize=(12, 8))

#     # Plot waveform
#     ax[0].set_title("Waveform")
#     librosa.display.waveshow(audio, sr=sr, ax=ax[0])

#     # Plot spectrogram
#     ax[1].set_title("Spectrogram")
#     X = librosa.stft(audio)
#     X_db = librosa.amplitude_to_db(abs(X))
#     img = librosa.display.specshow(X_db, sr=sr, x_axis='time', y_axis='hz', ax=ax[1])
#     fig.colorbar(img, ax=ax[1])

#     st.pyplot(fig)

# # Handle the "Upload" option
# if selected == "Upload":
#     uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

#     if uploaded_file:
#         # Display the uploaded audio file
#         st.audio(uploaded_file, format="audio/wav")

#         # Save uploaded file temporarily
#         temp_filename = "temp_audio.wav"
#         with open(temp_filename, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Load the audio file
#         audio, sr = librosa.load(temp_filename, sr=None)

#         # Extract MFCC features
#         mfcc_features = extract_features(temp_filename)

#         # Reshape MFCCs to match the input shape expected by the LSTM model
#         mfcc_features = np.reshape(mfcc_features, (1, mfcc_features.shape[0], 1))

#         # Make the prediction
#         prediction = model.predict(mfcc_features)
#         predicted_label = np.argmax(prediction)

#         # Display the predicted emotion
#         st.write(f"Predicted Emotion: **{emotion_dict[predicted_label]}**")

#         # Plot waveform and spectrogram
#         plot_waveform_and_spectrogram(audio, sr)

#         # Clean up temporary file
#         os.remove(temp_filename)

# # Handle the "Realtime" option (placeholder for future functionality)
# if selected == "Realtime":
#     st.write("Real-time emotion detection coming soon!")
#     # Add real-time functionality here later




import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from streamlit_option_menu import option_menu

# Load the LSTM model
model = load_model("models/emotion_detection_model.h5")

# Emotion mapping (same as during training)
emotion_dict = {
    0: 'Angry',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Neutral',
    5: 'Fearful',
    6: 'Disgust',
    7: 'Surprised',
    8: 'Pleasant Surprise'
}

# Configure Streamlit layout
st.set_page_config(
    page_title="Speech Emotion Detection",
    page_icon="assets/emotion_image.png",
    layout="wide",
)

# Sidebar for Image and additional content
with st.sidebar:
    st.image("assets/emotion_image.png", use_container_width=True)  
    

st.title("AudioXplore : Speech Emotion Detection")
st.markdown("Upload an audio file, and the model will identify the emotion of speech from the audio.")

# Option menu for selecting between "Upload"
selected = option_menu(
    None, 
    options=["Upload"],  # Removed "Realtime"
    icons=["file-earmark-music"],  # Updated icon for "Upload"
    menu_icon="cast",
    default_index=0,  # Default to "Upload"
    orientation="horizontal"
)

# Function to extract MFCC features from an audio file
def extract_features(filepath, n_mfcc=40):
    audio, sr = librosa.load(filepath, sr=None, res_type="kaiser_fast")
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Function to plot waveform and spectrogram
def plot_waveform_and_spectrogram(audio, sr):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Plot waveform
    ax[0].set_title("Waveform")
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])

    # Plot spectrogram
    ax[1].set_title("Spectrogram")
    X = librosa.stft(audio)
    X_db = librosa.amplitude_to_db(abs(X))
    img = librosa.display.specshow(X_db, sr=sr, x_axis='time', y_axis='hz', ax=ax[1])
    fig.colorbar(img, ax=ax[1])

    st.pyplot(fig)

# Handle the "Upload" option
if selected == "Upload":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_file:
        # Display the uploaded audio file
        st.audio(uploaded_file, format="audio/wav")

        # Save uploaded file temporarily
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the audio file
        audio, sr = librosa.load(temp_filename, sr=None)

        # Extract MFCC features
        mfcc_features = extract_features(temp_filename)

        # Reshape MFCCs to match the input shape expected by the LSTM model
        mfcc_features = np.reshape(mfcc_features, (1, mfcc_features.shape[0], 1))

        # Make the prediction
        prediction = model.predict(mfcc_features)
        predicted_label = np.argmax(prediction)

        # Display the predicted emotion
        st.write(f"Predicted Emotion: **{emotion_dict[predicted_label]}**")

        # Plot waveform and spectrogram
        plot_waveform_and_spectrogram(audio, sr)

        # Clean up temporary file
        os.remove(temp_filename)

