import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Conditional TensorFlow Import (Safe Mode) ---
TENSORFLOW_AVAILABLE = False
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Page Configuration
st.set_page_config(page_title="AutoEye Heart AI", page_icon="❤️", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp { color: #ffffff; }
    div.stButton > button { width: 100%; border-radius: 8px; background-color: #ff4b4b; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Resource Loading
@st.cache_resource
def load_all_models():
    models = {}
    scaler = None
    models_dir = 'models'
    
    if os.path.exists(models_dir):
        # Load Scikit-Learn Models
        for filename in os.listdir(models_dir):
            if filename.endswith('.joblib') and filename != 'scaler.joblib':
                name = filename.replace('.joblib', '').replace('_', ' ').title()
                models[name] = joblib.load(os.path.join(models_dir, filename))
        
        # Load LSTM (Conditional Load)
        if TENSORFLOW_AVAILABLE:
            lstm_path = os.path.join(models_dir, 'lstm_model.h5')
            if os.path.exists(lstm_path):
                models['Lstm'] = load_model(lstm_path)
            
        # Load Scaler
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            
    return models, scaler

models, scaler = load_all_models()

# --- UI Layout ---
st.title("❤️ AutoEye Heart AI Diagnostics")
st.markdown("#### Professional Clinical Decision Support System | Developed by Asad Riaz")
st.write("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
if not models:
    st.error("No models found! Please ensure your 'models/' folder is uploaded.")
    st.stop()

selected_model = st.sidebar.selectbox("Select ML Algorithm:", list(models.keys()))
st.sidebar.info(f"Active Engine: **{selected_model}**")

# Input Layout
st.subheader("🩺 Patient Health Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 90, 200, 120)

with col2:
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Prediction Logic
st.write("---")
if st.button("🚀 Analyze Patient Health"):
    features = np.array([[age, 1 if sex=="Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    if scaler:
        features = scaler.transform(features)
        
    model = models[selected_model]
    
    # Prediction logic (Safe Mode)
    if selected_model == 'Lstm' and TENSORFLOW_AVAILABLE:
        pred = (model.predict(features.reshape(1,1,13)) > 0.5).astype(int)[0][0]
    else:
        pred = model.predict(features)[0]

    if pred == 1:
        st.error("⚠️ HIGH RISK: Potential Heart Disease detected. Consult a Cardiologist immediately.")
    else:
        st.success("✅ LOW RISK: No significant indicators of heart disease found.")
