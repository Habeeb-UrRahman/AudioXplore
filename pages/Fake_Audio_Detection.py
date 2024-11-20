import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from streamlit_option_menu import option_menu
import os

# Function to extract MFCC features from audio file
def extract_mfcc_features(audio_path, n_mfcc=40, max_pad_len=174):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.flatten(), y, sr  # Also return y, sr for visualization
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# Function to load the scaler
def load_scaler():
    return joblib.load("models/scaler.pkl")

# Function to load the model
def load_audio_model():
    return load_model("models/fake_audio_detection_model.h5")

# Preprocess and predict the uploaded file
def predict_audio(audio_file, model, scaler):
    # Extract MFCC features
    mfcc_features, _, _ = extract_mfcc_features(audio_file)
    if mfcc_features is not None:
        # Scale the features using the scaler
        X = scaler.transform([mfcc_features])
        # Predict the label
        prediction_prob = model.predict(X).flatten()
        prediction_label = "Real" if prediction_prob < 0.5 else "Fake"
        return prediction_label, prediction_prob
    return None, None

# Plot MFCC for the uploaded audio
def plot_mfcc(audio_file):
    mfcc, y, sr = extract_mfcc_features(audio_file)
    if mfcc is not None:
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr)), 
                                                      sr=sr, n_mfcc=13), x_axis='time')
        plt.title("MFCC of Uploaded Audio")
        plt.colorbar(format="%+2.0f dB")
        st.pyplot(plt)

# Streamlit interface
# st.set_page_config(page_title="Audio Classification", layout="wide")
st.set_page_config(
    page_title="AI Audio Detection",
    page_icon="assets/fake_detect.png",
    layout="wide",
)
st.title("AudioXplore : AI Audio Detector")
st.markdown("""
Upload an audio file, and the model will classify whether the audio is real or fake.
""")

# Option menu for selecting between "Upload"
selected = option_menu(
    None, 
    options=["Upload"],  # Removed "Realtime"
    icons=["file-earmark-music"],  # Updated icon for "Upload"
    menu_icon="cast",
    default_index=0,  # Default to "Upload"
    orientation="horizontal"
)
# Add a sidebar for quick navigation
with st.sidebar:
    st.header("Navigation")
    st.markdown("Choose an audio file to analyze.")
    st.image("assets/fake_detect.png", use_container_width=True)  # Add a logo or image in the sidebar

# File uploader with a nice description
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg", "aac", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    audio_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display loading spinner while processing
    with st.spinner("Loading model and making predictions..."):
        # Load model and scaler
        model = load_audio_model()
        scaler = load_scaler()

        # Predict the uploaded audio file
        prediction_label, prediction_prob = predict_audio(audio_path, model, scaler)

        # Display the result with clear styling
        if prediction_label is not None:
            # Prediction result
            st.subheader("📊 Audio Classification Result")
            st.markdown(f"**Prediction**: The audio is classified as **{prediction_label}**")
            st.markdown(f"**Probability**: {prediction_prob[0]:.4f}")

            # Display MFCC plot
            st.subheader("🔊 Audio Features (MFCC)")
            plot_mfcc(audio_path)

            # # Additional evaluation (ROC curve, Precision-Recall curve)
            # st.subheader("📈 Evaluation Metrics")

            # # ROC Curve
            # fpr, tpr, _ = roc_curve([0, 1], [1 - prediction_prob, prediction_prob])  # Fake vs Real (just for demo)
            # roc_auc = auc(fpr, tpr)
            # plt.figure(figsize=(8, 6))
            # plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (area = {roc_auc:.2f})")
            # plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            # plt.legend(loc='lower right')
            # st.pyplot(plt)

            # # Precision-Recall Curve
            # precision_vals, recall_vals, _ = precision_recall_curve([0, 1], [1 - prediction_prob, prediction_prob])  # Fake vs Real
            # pr_auc = auc(recall_vals, precision_vals)
            # plt.figure(figsize=(8, 6))
            # plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f"Precision-Recall Curve (area = {pr_auc:.2f})")
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title('Precision-Recall Curve')
            # plt.legend(loc='lower left')
            # st.pyplot(plt)
        else:
            st.error("Error: Could not extract features from the uploaded audio.")
else:
    st.info("Please upload an audio file to begin the analysis.")
