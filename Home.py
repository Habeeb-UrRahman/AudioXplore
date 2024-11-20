import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AudioXplore",
    page_icon="assets/favicon.png",
    layout="wide",
)

# Injecting custom CSS for styling and animation
st.markdown("""
    <style>
        /* General Styling */
        body {
            margin: 0;
            padding: 0;
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #eef2f3, #ffffff);
            color: #333;
        }

        /* Hero Section */
        .hero {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 80px 20px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.3);
            margin-bottom: 30px;
        }
        .hero h1 {
            font-size: 102px;
            font-weight: bold;
            margin-bottom: 20px;
            animation: fadeInDown 1.2s ease;
        }
        .hero p {
            font-size: 18px;
            max-width: 800px;
            margin: 0 auto;
            animation: fadeInUp 1.2s ease;
        }

        /* Cards Section */
        .cards-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            padding: 40px 20px;
        }
        .card {
            background: #ffffff;
            border-radius: 22px;
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.1);
            padding: 50px;
            text-align: center;
            transition: transform 1s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0px 12px 25px rgba(0, 0, 0, 0.2);
        }
        .emoji {
            font-size: 60px;
            margin-bottom: 15px;
            animation: bounce 2s infinite;
        }
        .card h3 {
            font-size: 24px;
            color: #333;
            margin-bottom: 10px;
        }
        .card p {
            font-size: 16px;
            color: #555;
        }

        /* Animations */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        /* Footer Section */
        .footer {
            background-color: #2d3436;
            color: white;
            padding: 20px 0;
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
            border-top-left-radius: 12px;
            border-top-right-radius: 12px;
        }
        .footer a {
            color: #81ecec;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero">
        <h1>Welcome to AudioXplore!</h1>
        <p>
            
Explore tools designed for speech analysis, including speaker identification, fake audio detection, emotion analysis, and more.
        </p>
    </div>
""", unsafe_allow_html=True)

# Tools Section
st.markdown('<div class="cards-container">', unsafe_allow_html=True)

# Card 1: Speaker Identification
st.markdown("""
    <div class="card">
        <div class="emoji">👤</div>
        <h3>Speaker Identification</h3>
        <p>Identify who is speaking in an audio clip using advanced voice recognition algorithms.</p>
    </div>
""", unsafe_allow_html=True)

# Card 2: Fake Audio Detection
st.markdown("""
    <div class="card">
        <div class="emoji">🕵️‍♀️</div>
        <h3>Fake Audio Detection</h3>
        <p>Detect whether the audio is real or artificially generated, helping to spot deepfake content.</p>
    </div>
""", unsafe_allow_html=True)

# Card 3: Emotion Detection
st.markdown("""
    <div class="card">
        <div class="emoji">😡</div>
        <h3>Emotion Detection</h3>
        <p>Analyze the emotional tone of speech to determine feelings like joy, anger, or sadness.</p>
    </div>
""", unsafe_allow_html=True)

# Card 4: Gender & Age Detection
st.markdown("""
    <div class="card">
        <div class="emoji">👵</div>
        <h3>Gender & Age Detection</h3>
        <p>Estimate the gender and approximate age of a speaker based on their voice characteristics.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer Section
st.markdown("""
    <div class="footer">
        <p>
            Developed by <a href="https://www.linkedin.com/in/habeeb-urrahman/">Habeeb Ur Rahman</a> | <a href="https://www.linkedin.com/in/t-jaswanth-981214251/">Jaswanth Sri Nagababu</a> | <a href="https://www.linkedin.com/in/havish-ponnaganti-480b17226/">Havish Ponnaganti</a> | <a href="https://www.linkedin.com/in/rahul-reddy-b7a377227/">Bhimavarapu Rahul</a>
        </p>
    </div>
""", unsafe_allow_html=True)


