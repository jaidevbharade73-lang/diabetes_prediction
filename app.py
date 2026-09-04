import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Diabetes AI Predictor | Jaidev & Anmol",
    page_icon="🩺",
    layout="centered"
)

# 2. Clean Light Theme Styling
st.markdown("""
<style>
.stApp {
    background-color: #f4f6f9;
    color: #111827;
}

.header-tag {
    color: #2563eb;
    font-weight: 700;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# 3. Train Model
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

# 4. Session State Management
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'user_data' not in st.session_state:
    st.session_state.user_data = None


# --- PAGE 1: WELCOME SCREEN ---
if st.session_state.page == 'welcome':
    st.markdown('<p class="header-tag">🤖 AI MODEL DEVELOPED BY JAIDEV AND ANMOL</p>', unsafe_allow_html=True)
    st.title("🩺 Diabetes Prediction Web Application")
    
    st.info("Welcome! This interactive machine learning tool evaluates clinical patient parameters to predict diabetes risk instantly.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➡️ Click Next to Enter Details", use_container_width=True, type="primary"):
        st.session_state.page = 'input'
        st.rerun()


# --- PAGE 2: INPUT FORM SCREEN (FULL-WIDTH BROAD FIELDS) ---
elif st.session_state.page == 'input':
    st.markdown('<p class="header-tag">🤖 AI MODEL DEVELOPED BY JAIDEV AND ANMOL</p>', unsafe_allow_html=True)
    
    if st.button("⬅️ Back"):
        st.session_state.page = 'welcome'
        st.rerun()
        
    st.title("📋 Enter Patient Medical Parameters")
    st.write("Fill in all patient parameters below to run the AI prediction.")
    st.markdown("---")
    
    # All fields listed sequentially to make each input box full-width across the page
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI Value", min_value=0.0, max_value=70.0, value=25.00, format="%.2f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.500, format="%.3f")
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=30)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Run Prediction Model", use_container_width=True, type="primary"):
        st.session_state.user_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        st.session_state.page = 'results'
        st.rerun()


# --- PAGE 3: LOADING & RESULTS SCREEN ---
elif st.session_state.page == 'results':
    st.markdown('<p class="header-tag">🤖 AI MODEL DEVELOPED BY JAIDEV AND ANMOL</p>', unsafe_allow_html=True)
    st.title("📊 Diagnostic Assessment")
    
    with st.spinner("⏳ Analyzing patient data and calculating risk model... Please wait..."):
        time.sleep(3)
    
    data = st.session_state.user_data
    input_list = [[
        data['Pregnancies'], data['Glucose'], data['BloodPressure'],
        data['SkinThickness'], data['Insulin'], data['BMI'],
        data['DiabetesPedigreeFunction'], data['Age']
    ]]
    
    prediction = model.predict(input_list)
    probability = model.predict_proba(input_list)[0][1] * 100
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ **Result: High Risk of Diabetes Detected** ({probability:.1f}% risk score)")
    else:
        st.success(f"✅ **Result: Low Risk of Diabetes Detected** ({probability:.1f}% risk score)")
        
    st.markdown("### Submitted Patient Summary")
    st.json(data)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Test Another Patient", use_container_width=True):
        st.session_state.page = 'input'
        st.rerun()
    



        
