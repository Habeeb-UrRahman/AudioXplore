import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import io
from streamlit_option_menu import option_menu

# Load the trained models
gender_model = tf.keras.models.load_model('models/gender_classification_model.h5')
male_age_model = tf.keras.models.load_model('models/male_age_model.h5')
female_age_model = tf.keras.models.load_model('models/female_age_model.h5')

# Define MFCC extraction function
def extract_features(uploaded_file, target_length=130):
    try:
        audio_data = uploaded_file.getvalue()
        y, sr = librosa.load(io.BytesIO(audio_data), sr=16000)

        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        if log_spectrogram.shape[1] < target_length:
            pad_width = target_length - log_spectrogram.shape[1]
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        elif log_spectrogram.shape[1] > target_length:
            log_spectrogram = log_spectrogram[:, :target_length]

        return log_spectrogram
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Predict gender
def predict_gender(features):
    features = features[np.newaxis, ..., np.newaxis]
    predictions = gender_model.predict(features)
    gender = 'male' if np.argmax(predictions) == 0 else 'female'
    confidence = predictions[0][np.argmax(predictions)]
    return gender, confidence

# Predict age group
def predict_age(features, model, gender):
    age_groups = ["child", "teen", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]
    features = features[np.newaxis, ..., np.newaxis]
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions)
    predicted_class = age_groups[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence

# Streamlit UI Configuration
st.set_page_config(
    page_title="Gender & Age Identification",
    page_icon="assets/age_icon.png",
    layout="wide",
)

# Sidebar with image
with st.sidebar:
    st.image("assets/age_icon.png", use_container_width=True)
    st.write("AudioXplore: Analyze gender and age group from audio clips!")

# Header
st.title("AudioXplore: Gender and Age Prediction System")
st.markdown("Upload an audio file to predict the gender and age group of the speaker!")

# Horizontal menu
selected = option_menu(
    None, 
    options=["Upload"], 
    icons=["file-earmark-music"], 
    menu_icon="cast", 
    default_index=0,
    orientation="horizontal"
)

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (wav or mp3)", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    st.markdown("**Processing your file...**")

    # Identify button with loader
    if st.button('Analyze Audio', key="analyze_button"):
        with st.spinner("Analyzing audio..."):
            features = extract_features(uploaded_file)
            if features is not None:
                gender, gender_confidence = predict_gender(features)
                if gender == 'male':
                    age_group, age_confidence = predict_age(features, male_age_model, gender)
                else:
                    age_group, age_confidence = predict_age(features, female_age_model, gender)

                st.success(f"🎤 **Predicted Gender:** {gender} ({gender_confidence*100:.2f}% confidence)")
                st.success(f"📅 **Predicted Age Group:** {age_group} ({age_confidence*100:.2f}% confidence)")
            else:
                st.error("⚠️ Unable to extract features from the audio file. Please try again.")
