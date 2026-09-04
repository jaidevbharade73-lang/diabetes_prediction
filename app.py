import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
    page_title="Diabetes AI Predictor | Jaidev & Anmol",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for Background, Fonts, and Glassmorphism Cards
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Modern Dark Blue Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0b132b 0%, #1c2541 50%, #3a506b 100%);
    color: #ffffff;
}

/* Hero Welcome Card */
.welcome-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 35px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-top: 15px;
    margin-bottom: 25px;
}

.welcome-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: #4cc9f0;
    margin-top: 10px;
    margin-bottom: 10px;
}

.author-badge {
    background: linear-gradient(90deg, #4895ef, #4361ee);
    color: white;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    display: inline-block;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #4cc9f0 0%, #4361ee 100%);
    color: white;
    border: none;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 30px;
    transition: all 0.3s ease;
    width: 100%;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(76, 201, 240, 0.4);
    color: white;
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
if st.session_state.page == 'welcome':
    st.markdown("""
        <div class="welcome-card">
            <span class="author-badge">AI MODEL DEVELOPED BY JAIDEV & ANMOL</span>
            <div class="welcome-title">🩺 Diabetes Prediction AI</div>
            <p style="color: #cbd5e1; font-size: 1.05rem; line-height: 1.6;">
                Welcome! This interactive machine learning tool evaluates clinical patient parameters to predict diabetes risk instantly.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Click Next to Predict ➔"):
            st.session_state.page = 'predict'
            st.rerun()

# --- PAGE 2: PREDICTION INPUT FORM ---
elif st.session_state.page == 'predict':
    if st.button("⬅️ Back to Welcome Screen"):
        st.session_state.page = 'welcome'
        st.rerun()
        
    st.markdown("<h2 style='color: #4cc9f0; text-align: center; margin-top: 15px;'>Patient Medical Parameters</h2>", unsafe_allow_html=True)
    st.caption("Enter the required medical details below to run the AI diagnosis model.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=79)
        bmi = st.number_input("BMI Value", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

    user_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Run Prediction Model"):
        prediction = model.predict(user_data)
        if prediction[0] == 1:
            st.error("⚠️ **Result: High Risk of Diabetes Detected.**")
        else:
            st.success("✅ **Result: Low Risk of Diabetes Detected.**")

