import streamlit as st
import librosa
import numpy as np
import joblib
import io
import tensorflow as tf
from streamlit_option_menu import option_menu

# Load the trained model and label encoder
model = tf.keras.models.load_model('models/speaker_recognition_model.h5')
label_encoder = joblib.load('models/label_encoder.pkl')

# Define the MFCC extraction function
def extract_mfcc(uploaded_file):
    try:
        # Load audio file from uploaded byte stream
        audio_data = uploaded_file.getvalue()
        y, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Ensure the MFCC array has more than one time frame (shape[1] > 0)
        if mfcc.shape[1] == 0:  # Check if the MFCC matrix has a time axis
            return None, None, None
        
        return np.mean(mfcc.T, axis=0), y, sr
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None, None, None

# Prediction function
def predict_speaker(audio_file, model, label_encoder):
    mfcc, y, sr = extract_mfcc(audio_file)
    if mfcc is None:
        return None
    
    # Reshape MFCC to match the input shape of the model
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch dimension and channel dimension
    pred = model.predict(mfcc)
    
    predicted_label = np.argmax(pred)
    return label_encoder.inverse_transform([predicted_label])[0]

# Streamlit UI Enhancements
st.set_page_config(
    page_title="Speaker Identification System",
    page_icon="assets/speaker_image.png",
    layout="wide",
)

# Sidebar for Image and additional content
with st.sidebar:
    st.image("assets/speaker_image.png", use_container_width=True)  # Example image

# Layout for the app header without columns
st.title("AudioXplore : Speaker Identification System")
st.markdown("Upload an audio file, and our model will identify the speaker!")

# Option menu for selecting between "Upload"
selected = option_menu(
    None, 
    options=["Upload"],  # Removed "Realtime"
    icons=["file-earmark-music"],  # Updated icon for "Upload"
    menu_icon="cast",
    default_index=0,  # Default to "Upload"
    orientation="horizontal"
)

# Full-width layout for the app content
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")  # Display the audio player
    st.markdown("**Note:** Please upload a `.wav` file for accurate speaker identification.")

# Button to trigger speaker identification
if uploaded_file is not None:
    if st.button('Identify Speaker', key="identify_button"):
        # Show progress indicator while processing the file
        with st.spinner("Processing audio..."):
            # Predict speaker
            speaker = predict_speaker(uploaded_file, model, label_encoder)
            
            if speaker:
                st.success(f"🎤 **Predicted Speaker**: {speaker}")
            else:
                st.error("⚠️ Failed to predict the speaker. Please check the file and try again.")
