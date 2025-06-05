# AudioXplore 🎧 | Advanced Integrated Audio Analysis System

**AudioXplore** is a deep learning–powered audio analysis system that combines multiple AI tools into a single interactive platform. It enables real-time detection and classification of speaker identity, AI-generated audio, speaker emotion, and gender/age prediction from voice recordings.

🎙️ **Live Demo:** [https://audioxplore.streamlit.app/](https://audioxplore.streamlit.app/)

---

## 🧠 About the Project

AudioXplore is built to serve as an intelligent audio forensics and voice profiling tool. It extracts MFCC features from voice clips — a compact representation of how humans perceive sound — and runs them through specialized deep learning models for each classification task.

### 🔍 Core Features

- **🗣️ Speaker Identification**
  - Recognizes individual speakers (e.g., celebrities or known voices).
  - Built using **MFCC + Conv2D CNN** architecture.
  - Achieved **97% accuracy** on curated datasets.

- **🤖 AI-Generated Audio Detection**
  - Detects synthetic voices and deepfakes.
  - Uses MFCC + CNN classifiers with a focus on adversarial training.
  - Reached **95% detection accuracy**.

- **🎭 Speaker Emotion Detection**
  - Classifies emotions such as happy, sad, angry, or neutral.
  - Achieved **80% accuracy** across emotion-labeled speech corpora.

- **⚧️ Gender and Age Classification**
  - Predicts speaker’s gender and groups age (e.g., child, adult, senior).
  - Achieved **88% accuracy** with multi-head CNN architecture.

---

## 📈 Model Architectures & Techniques

- **Feature Extraction:** MFCC (Mel-Frequency Cepstral Coefficients)
- **Speaker & Fake Audio Detection:** Conv2D CNN
- **Emotion Detection:** CNN with dense layers
- **Age/Gender Prediction:** Multi-output CNN with softmax classification heads

---

## 🛠 Technologies Used

- **Deep Learning:** TensorFlow, Keras
- **Audio Processing:** Librosa, NumPy, SoundFile
- **Frontend:** Streamlit (UI built with Python)
- **Deployment:** Streamlit Cloud

---

## 📂 Project Structure

