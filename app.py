import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
    page_title="Diabetes AI Predictor | Jaidev & Anmol",
    page_icon="🩺",
    layout="centered"
)

# High-Contrast Light Medical Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Light Gradient Background for Maximum Readability */
.stApp {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    color: #0f172a;
}

/* Card Boxes */
.card-box {
    background: #ffffff;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    border: 1px solid #cbd5e1;
    margin-bottom: 20px;
}

/* Super Clear Dark Labels for Inputs */
label, .stMarkdown, p {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Headings */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0284c7;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #334155 !important;
    font-weight: 400 !important;
    margin-bottom: 20px;
}

.author-tag {
    background: #0284c7;
    color: white !important;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 15px;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #0284c7 0%, #0369a1 100%);
    color: white !important;
    border: none;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 12px;
    width: 100%;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(2, 132, 199, 0.4);
}
</style>
""", unsafe_allow_html=True)

# Train the Model
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)
    
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = load_and_train()

# Page Navigation State
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# --- PAGE 1: WELCOME SCREEN ---

        
