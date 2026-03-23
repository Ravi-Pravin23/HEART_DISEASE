import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import os
import hashlib
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import requests

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_resource
def load_models():
    """Loads the AI models and scalers securely."""
    return joblib.load('models/heart_model.pkl'), joblib.load('models/scaler.pkl'), joblib.load('models/features.pkl')

def create_pdf(name, pred, probability, age_val, chol_val, bp_val, weight_val):
    """Generates a downloadable PDF report for the patient."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15, style="B")
    pdf.cell(200, 10, txt="Heart AI Diagnostic Medical Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, txt="Patient Information:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Assessed by: Dr. {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age_val} | Weight: {weight_val} kg | Chol: {chol_val} mg/dl | BP: {bp_val} mmHg", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, txt="Diagnostic Results:", ln=True)
    pdf.set_font("Arial", size=12)
    disease_map = {
        0: 'Healthy / Low Risk',
        1: 'Coronary Artery Disease (CAD)',
        2: 'Heart Attack (Myocardial Infarction)',
        3: 'Arrhythmia',
        4: 'Heart Failure (Congestive)',
        5: 'Heart Valve Disease',
        6: 'Cardiomyopathy',
        7: 'Congenital Heart Defects',
        8: 'Pericarditis',
        9: 'Myocarditis',
        10: 'Endocarditis',
        11: 'Aortic Aneurysm',
        12: 'Peripheral Artery Disease (PAD)'
    }
    status = disease_map.get(pred, 'Unknown')
    pdf.cell(200, 10, txt=f"Prediction: {status}", ln=True)
    pdf.cell(200, 10, txt=f"Diagnostic Confidence: {probability*100:.2f}%", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=10, style="I")
    pdf.cell(200, 10, txt="Note: This is an AI-generated report. Please consult a doctor for a formal clinical diagnosis.", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 1. DATABASE SETUP (FR-1: User Registration) ---
DB_NAME = "data/users.db"
PATIENTS_DB = "data/patients.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def init_patients_db():
    conn = sqlite3.connect(PATIENTS_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_name TEXT,
        age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER, chol INTEGER,
        fbs INTEGER, restecg INTEGER, thalach INTEGER, exang INTEGER,
        oldpeak REAL, slope INTEGER, ca INTEGER, thal INTEGER, weight INTEGER,
        prediction_str TEXT, probability REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Simple migration: add weight column if it doesn't exist
    try:
        c.execute("ALTER TABLE patients ADD COLUMN weight INTEGER DEFAULT 70")
    except sqlite3.OperationalError:
        pass # Already exists
    
    conn.commit()
    conn.close()

def save_patient_record(doctor, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, weight, pred_str, prob):
    conn = sqlite3.connect(PATIENTS_DB)
    c = conn.cursor()
    c.execute('''INSERT INTO patients (
        doctor_name, age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal, weight, prediction_str, probability
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
    (doctor, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, weight, pred_str, prob))
    conn.commit()
    conn.close()

def load_users():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, password FROM users")
    users = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return users

def save_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("REPLACE INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()
    conn.close()

def migrate_json_to_sqlite():
    if os.path.exists("data/users.json"):
        with open("data/users.json", "r") as f:
            try:
                old_users = json.load(f)
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                for uname, pwd in old_users.items():
                    c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", (uname, pwd))
                conn.commit()
                conn.close()
                os.rename("data/users.json", "data/users.json.bak")
            except Exception:
                pass

init_db()
init_patients_db()
migrate_json_to_sqlite()

# --- 2. SESSION STATE (FR-2: Authentication) ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ""

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="Heart AI Clinical Portal", page_icon="⚕️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Force light color scheme for all browsers */
    :root {
        color-scheme: light !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1a202c !important;
    }

    /* Force light background on the entire app */
    .stApp {
        background-color: #f5f7fa !important;
    }

    /* Sidebar: force white */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #1a202c !important;
    }

    /* All text inputs: force light background */
    input, textarea, [data-baseweb="input"] input,
    [data-baseweb="base-input"] {
        background-color: #ffffff !important;
        color: #1a202c !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
    }

    /* Selectbox / dropdown */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1a202c !important;
        border: 1px solid #d1d5db !important;
    }

    /* Labels */
    .stTextInput label, .stSelectbox label, .stSlider label,
    .stNumberInput label, label, p, span, h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
    }

    /* Tabs: force readable text */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        font-weight: 500;
        color: #4b5563 !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1a56db !important;
        font-weight: 600;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #1a56db !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
        font-weight: 500;
    }

    /* Primary action buttons */
    .stButton > button {
        background-color: #1a56db !important;
        color: white !important;
        border: none !important;
        border-radius: 6px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1e429f !important;
        color: white !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #047857 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background-color: #065f46 !important;
        color: white !important;
    }

    /* Force main content area light */
    [data-testid="stMain"], [data-testid="block-container"] {
        background-color: #f5f7fa !important;
    }

    /* Expander header text */
    [data-testid="stExpander"] summary span {
        color: #1a202c !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. AUTHENTICATION UI (Login / Register) ---
if not st.session_state['logged_in']:
    st.markdown("")
    st.markdown("")
    st.markdown("<h1 style='text-align:center; color:#1a202c; font-weight:700;'>Heart AI Clinical Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.05rem; color:#6b7280; margin-bottom:2rem;'>Advanced Diagnostic Intelligence for Cardiology Professionals</p>", unsafe_allow_html=True)
    st.write("---")
    
    _, auth_col, _ = st.columns([1, 2, 1])
    
    with auth_col:
        auth_tab1, auth_tab2 = st.tabs(["🔒 Login", "📝 Register"])
        
        with auth_tab2:
            st.markdown("### Create New Account")
            new_user = st.text_input("Username", key="reg_u")
            new_pass = st.text_input("Password", type="password", key="reg_p")
            if st.button("Register Now", use_container_width=True):
                if new_user and new_pass:
                    save_user(new_user, new_pass)
                    st.success("Account created! You can now login.")
                else:
                    st.warning("Fields cannot be empty.")

        with auth_tab1:
            st.markdown("### Login to Dashboard")
            log_user = st.text_input("Username", key="log_u")
            log_pass = st.text_input("Password", type="password", key="log_p")
            if st.button("Login", use_container_width=True):
                users = load_users()
                # Support both old (plain-text) and new (hashed) passwords for existing users
                if log_user in users and (users[log_user] == log_pass or users[log_user] == hash_password(log_pass)):
                    # Convert plain-text to hashed password on-the-fly upon login
                    if users[log_user] == log_pass:
                        save_user(log_user, log_pass)
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = log_user
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

# --- 5. MAIN APPLICATION (Visible after Login) ---
else:
    # --- Sidebar Navigation Menu ---
    st.sidebar.title(f"Dr. {st.session_state['user_name']}")
    st.sidebar.markdown("---")
    
    menu_selection = st.sidebar.radio(
        "Navigation Menu",
        ["🩺 Diagnostic Assessment", "📈 Patient Dashboard", "📂 Patient History", "📊 Data Explorer"]
    )
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("⚙️ n8n Integration", expanded=False):
        n8n_webhook_url = st.text_input(
            "Webhook URL",
            value="http://localhost:5678/webhook/heart-alert",  # Production URL (after publishing)
            help="n8n webhook that receives patient data and sends the email."
        )
        if st.button("Test Connection", use_container_width=True):
            try:
                res = requests.post(n8n_webhook_url, json={"test": True}, timeout=3)
                if res.status_code == 200:
                    st.success("✅ Connected to n8n!")
                else:
                    st.error(f"Reached n8n but got status: {res.status_code}")
            except Exception:
                st.error("❌ Could not reach n8n. Make sure it's running.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state['logged_in'] = False
        st.rerun()

    # ==========================================
    # PAGE 1: 🩺 DIAGNOSTIC ASSESSMENT
    # ==========================================
    if menu_selection == "🩺 Diagnostic Assessment":
        st.title("Heart Disease Diagnostic Assistant")
        st.write("Specialist Access:", st.session_state['user_name'])
        st.info("**Clinical AI Scope:** This newly upgraded multi-class model detects and differentiates between Coronary Artery Disease (CAD), Arrhythmias, and Heart Failure indicators.")
        st.markdown("---")

        # --- BATCH PROCESSING FEATURE ---
        with st.expander("📁 Batch Processing (Upload CSV)", expanded=False):
            st.markdown("Upload a CSV file containing multiple patient records to run batch AI diagnostics.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(batch_df)} patient records.")
                    if st.button("Run Batch Diagnostics", use_container_width=True):
                        try:
                            model, scaler, features = load_models()
                            if not all(col in batch_df.columns for col in features):
                                st.warning(f"CSV must contain the following columns: {features}")
                            else:
                                batch_input = batch_df[features].values
                                scaled_batch = scaler.transform(batch_input)
                                
                                batch_preds = model.predict(scaled_batch)
                                batch_probs = np.max(model.predict_proba(scaled_batch), axis=1)
                                
                                disease_map = {
                                    0: 'Healthy / Low Risk', 1: 'Coronary Artery Disease (CAD)', 2: 'Heart Attack (Myocardial Infarction)',
                                    3: 'Arrhythmia', 4: 'Heart Failure (Congestive)', 5: 'Heart Valve Disease',
                                    6: 'Cardiomyopathy', 7: 'Congenital Heart Defects', 8: 'Pericarditis',
                                    9: 'Myocarditis', 10: 'Endocarditis', 11: 'Aortic Aneurysm', 12: 'Peripheral Artery Disease (PAD)'
                                }
                                
                                batch_df['AI Diagnosis'] = [disease_map.get(int(p), 'Unknown') for p in batch_preds]
                                batch_df['Confidence (%)'] = (batch_probs * 100).round(2)
                                
                                st.success("Batch processing complete!")
                                st.dataframe(batch_df[['AI Diagnosis', 'Confidence (%)'] + features].head(10), use_container_width=True)
                                
                                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Scored Batch Report (CSV)",
                                    data=csv_data,
                                    file_name="Batch_Diagnostics_Report.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Error during batch processing: {e}. Ensure models are trained using 'python src/train.py'.")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        st.markdown("---")

    # FR-3: Accept Patient Health Data
    st.markdown("### Single Patient Clinical Profile")
    st.markdown("Enter patient metrics below for AI risk stratification.")
    
    patient_email = st.text_input("Patient Contact Email (For Alerts)", value="ravipravin2005@gmail.com", help="This email will receive automated alerts if the AI detects a high-risk profile.")
    st.markdown("---")
    
    col_demo, col_vitals = st.columns(2)
    with col_demo:
        st.markdown("##### 1. Patient Demographics")
        age = st.number_input("Age (Years)", min_value=1, max_value=100, value=45)
        sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        weight = st.number_input("Weight (kg)", min_value=10, max_value=250, value=70)
        
    with col_vitals:
        st.markdown("##### 2. Vital Signs")
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 250, 120)
        thalach = st.slider("Maximum Heart Rate Achieved", 50, 250, 150)
        
    st.markdown("##### 3. Laboratory & ECG Results")
    col_lab1, col_lab2, col_lab3 = st.columns(3)
    with col_lab1:
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col_lab2:
        restecg = st.selectbox("Resting ECG Result", [0, 1, 2], help="0: Normal, 1: ST-T wave abnormal, 2: LV hypertrophy")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal, 3: Asymptomatic")
    with col_lab3:
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.slider("ST Depression (Exercise vs Rest)", 0.0, 10.0, 1.0, step=0.1)

    st.markdown("##### 4. Specialized Assessments")
    col_spec1, col_spec2, col_spec3 = st.columns(3)
    with col_spec1:
        slope = st.selectbox("Peak Exercise ST Segment Slope", [0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping")
    with col_spec2:
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    with col_spec3:
        thal = st.selectbox("Thalassemia Assessment", [1, 2, 3], help="1: Normal, 2: Fixed defect, 3: Reversable defect")

    st.markdown("---")
    # AI Prediction Trigger
    if st.button("RUN FULL DIAGNOSTIC ANALYSIS", use_container_width=True):
        # Load Model
        try:
            model, scaler, features = load_models()
        except:
            st.error("Error: Model files missing. Please run 'python src/train.py' first.")
            st.stop()

        # Data Processing
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, weight]])
        scaled_input = scaler.transform(input_data)
        prediction = int(model.predict(scaled_input)[0])
        prob = float(model.predict_proba(scaled_input)[0][prediction])

        disease_map = {
            0: 'Healthy / Low Risk',
            1: 'Coronary Artery Disease (CAD)',
            2: 'Heart Attack (Myocardial Infarction)',
            3: 'Arrhythmia',
            4: 'Heart Failure (Congestive)',
            5: 'Heart Valve Disease',
            6: 'Cardiomyopathy',
            7: 'Congenital Heart Defects',
            8: 'Pericarditis',
            9: 'Myocarditis',
            10: 'Endocarditis',
            11: 'Aortic Aneurysm',
            12: 'Peripheral Artery Disease (PAD)'
        }
        disease_name = disease_map.get(prediction, 'Unknown')

        # --- Save to Patient History Database ---
        save_patient_record(
            st.session_state['user_name'], age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal, weight, disease_name, prob
        )

        # --- Send to n8n → n8n emails the patient ---
        if patient_email:
            try:
                payload = {
                    "patient_email": patient_email,
                    "patient_age": age,
                    "patient_weight": weight,
                    "blood_pressure": trestbps,
                    "cholesterol": chol,
                    "risk_probability": f"{prob*100:.2f}%",
                    "disease_type": disease_name,
                    "diagnosed_by": st.session_state['user_name']
                }
                response = requests.post(n8n_webhook_url, json=payload, timeout=4)
                if response.status_code == 200:
                    st.success(f"✅ Diagnostic report sent to **{patient_email}** via n8n!")
                else:
                    st.warning(f"⚠️ n8n responded with status {response.status_code}")
            except Exception:
                st.info("💡 n8n not reachable — configure the webhook URL in the sidebar.")


        st.markdown("---")
        # --- Metrics Dashboard ---
        st.markdown("### Vital Signs Snapshot")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cholesterol", f"{chol} mg/dl", delta="High" if chol > 200 else "Normal", delta_color="inverse" if chol > 200 else "normal")
        m2.metric("Resting BP", f"{trestbps} mmHg", delta="High" if trestbps > 130 else "Normal", delta_color="inverse" if trestbps > 130 else "normal")
        m3.metric("Max Heart Rate", f"{thalach} bpm")
        
        st.markdown("---")
        
        # --- Radar Chart for Patient Profile ---
        st.markdown("### Patient Health Profile vs Healthy Baseline")
        categories = ['Age', 'Resting BP', 'Cholesterol', 'Max HR', 'ST Depression', 'Weight']
        
        # Normalize the values visually for the radar chart (rough scaling to 0-100 for comparison)
        patient_values = [
            min(100, (age / 100) * 100), 
            min(100, (trestbps / 200) * 100), 
            min(100, (chol / 400) * 100), 
            min(100, (thalach / 200) * 100), 
            min(100, (oldpeak / 6) * 100),
            min(100, (weight / 150) * 100)
        ]
        healthy_baseline = [45, 60, 50, 70, 10, 46.7] # 46.7 is ~70kg normalized
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
              r=patient_values,
              theta=categories,
              fill='toself',
              name='Patient vitals',
              line_color='#0A74DA', # Clinical Blue
              fillcolor='rgba(10, 116, 218, 0.2)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
              r=healthy_baseline,
              theta=categories,
              fill='toself',
              name='Healthy Baseline',
              line_color='#2ecc71', # Healthy Green
              fillcolor='rgba(46, 204, 113, 0.2)'
        ))
        
        fig_radar.update_layout(
          polar={"radialaxis": {"visible": True, "range": [0, 100], "showticklabels": False}},
          showlegend=True,
          margin={"t": 30, "b": 30, "l": 30, "r": 30},
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        res_col, advice_col = st.columns([1, 1])

        # --- FR-5: Result Display ---
        with res_col:
            st.subheader("Diagnostic Summary")
            if prediction > 0:
                st.error(f"### RESULT: {disease_name}")
            else:
                st.success(f"### RESULT: {disease_name}")
            
            # --- Interactive Risk Gauge ---
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Diagnostic Confidence (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if prediction > 0 else "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # --- AI Explainability (Top Factors) ---
            st.write("---")
            st.write("**AI Decision Factors (Why this result?)**")
            importances = model.feature_importances_
            feat_importances = pd.Series(importances, index=features).nlargest(5)
            
            fig_bar = px.bar(
                x=feat_importances.values, 
                y=feat_importances.index, 
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                color_discrete_sequence=['#0056b3']
            )
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Personalized Advice & Diet (Food Update) ---
        with advice_col:
            st.subheader("Personalized Health & Diet Plan")
            
            advice = {
                0: ("st.info",  "**Maintenance Plan:**",
                    "- **Diet:** Continue high-fiber foods like **Oats and Almonds**.\n- **Exercise:** Target 150 mins of moderate cardio per week."),
                1: ("st.warning", "**CAD Action Plan:**",
                    "- **Diet:** Switch to Mediterranean diet. Focus on **Olive Oil, Spinach, and Fatty Fish**.\n- **Avoid:** Red meat and processed snacks.\n- **Next Step:** Schedule a Cardiac Stress Test and Lipid Profile."),
                2: ("st.error", "🚨 **Heart Attack Action Plan:**",
                    "- **Immediate:** Seek emergency care. Take aspirin if prescribed.\n- **Diet:** Strict low-fat, low-sodium diet.\n- **Next Step:** Urgent ECG, Troponin test, and Echocardiogram."),
                3: ("st.warning", "**Arrhythmia Action Plan:**",
                    "- **Diet:** Limit caffeine and alcohol significantly.\n- **Avoid:** Stress and sleep deprivation.\n- **Next Step:** 24-hour Holter Monitor to capture irregular rhythms."),
                4: ("st.error", "🚨 **Heart Failure Action Plan:**",
                    "- **Diet:** Strict low-sodium diet (<1500mg/day) and fluid restriction.\n- **Avoid:** Heavy exertion.\n- **Next Step:** Immediate Echocardiogram to assess ejection fraction."),
                5: ("st.warning", "**Heart Valve Disease Action Plan:**",
                    "- **Diet:** Heart-healthy diet, limit saturated fats.\n- **Avoid:** Intense physical exertion.\n- **Next Step:** Echocardiogram to evaluate valve function and severity."),
                6: ("st.warning", "**Cardiomyopathy Action Plan:**",
                    "- **Diet:** Low-sodium, low-fat diet. Avoid alcohol.\n- **Avoid:** Strenuous activity.\n- **Next Step:** Cardiac MRI and genetic counseling."),
                7: ("st.warning", "**Congenital Heart Defect Action Plan:**",
                    "- **Diet:** Balanced, nutrient-rich diet.\n- **Monitor:** Regular cardiac check-ups.\n- **Next Step:** Echocardiogram and referral to a Congenital Heart Specialist."),
                8: ("st.warning", "**Pericarditis Action Plan:**",
                    "- **Rest:** Limit physical activity until inflammation resolves.\n- **Medication:** NSAIDs (e.g., Ibuprofen) as prescribed.\n- **Next Step:** Echo and inflammatory marker blood tests (CRP, ESR)."),
                9: ("st.warning", "**Myocarditis Action Plan:**",
                    "- **Rest:** Strict activity restriction.\n- **Diet:** Anti-inflammatory foods — berries, leafy greens, fish.\n- **Next Step:** Cardiac MRI and Troponin/BNP blood tests."),
                10: ("st.error", "🚨 **Endocarditis Action Plan:**",
                    "- **Urgent:** IV antibiotic therapy required immediately.\n- **Monitor:** Blood cultures and echocardiogram.\n- **Next Step:** Immediate hospital admission and infectious disease consult."),
                11: ("st.error", "🚨 **Aortic Aneurysm Action Plan:**",
                    "- **Diet:** Low-sodium, blood-pressure-friendly diet.\n- **Avoid:** Heavy lifting and extreme exertion.\n- **Next Step:** CT Angiography to measure aneurysm size and monitor growth."),
                12: ("st.warning", "**PAD Action Plan:**",
                    "- **Diet:** Mediterranean diet. Increase omega-3 intake (Fish, Flaxseed).\n- **Exercise:** Supervised walking program.\n- **Next Step:** Ankle-Brachial Index (ABI) test and Doppler Ultrasound.")
            }

            fn, title, tips = advice.get(prediction, advice[0])
            getattr(st, fn.split('.')[-1])(title)
            st.markdown(tips)
            
            # Specific Clinical Advice
            st.write("---")
            if chol > 240:
                st.info("**Cholesterol Tip:** Eat more **Walnuts and Flaxseeds** to help naturally lower LDL levels.")
            if trestbps > 140:
                st.info("**Blood Pressure Tip:** Reduce salt intake and eat **Bananas** for potassium.")
            if weight > 90:
                st.info("**Weight Tip:** Excessive weight can strain the heart. Focus on a balanced calorie-deficit diet and daily moderate walking.")

        pdf_bytes = create_pdf(st.session_state['user_name'], prediction, prob, age, chol, trestbps, weight)
        st.download_button(
            label="Download Professional Medical Report (PDF)",
            data=pdf_bytes,
            file_name="Heart_AI_Report.pdf",
            mime="application/pdf"
        )

    # ==========================================
    # PAGE 2: 📈 PATIENT DASHBOARD
    # ==========================================
    elif menu_selection == "📈 Patient Dashboard":
        st.title("Clinic Overview Dashboard")
        st.markdown("View aggregate patient statistics and clinic activity.")
        st.markdown("---")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Patients Screened", "1,204", "+15 this week")
        c2.metric("High-Risk Diagnoses", "241", "+3 this week")
        c3.metric("Average Waiting Time", "12 mins", "-2 mins")
        
        st.markdown("---")
        st.subheader("Monthly Screening Volume")
        # Dummy data for the chart
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        screenings = [120, 150, 180, 170, 210, 190]
        fig_line = px.line(x=months, y=screenings, labels={'x': 'Month', 'y': 'Screenings'}, markers=True)
        fig_line.update_traces(line_color='#1a56db', marker_color='#1a56db')
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)

    # ==========================================
    # PAGE 3: 📂 PATIENT HISTORY
    # ==========================================
    elif menu_selection == "📂 Patient History":
        st.title("Patient Diagnostic History")
        st.markdown(f"View historical AI assessments conducted by **Dr. {st.session_state['user_name']}**.")
        st.markdown("---")
        
        conn = sqlite3.connect(PATIENTS_DB)
        history_df = pd.read_sql_query(
            "SELECT timestamp, age, sex, weight, trestbps AS 'BP', chol AS 'Cholesterol', prediction_str AS 'Diagnosis', probability AS 'Confidence (%)' FROM patients WHERE doctor_name = ? ORDER BY timestamp DESC", 
            conn, params=(st.session_state['user_name'],)
        )
        conn.close()
        
        if history_df.empty:
            st.info("No patient history found. Run a diagnostic assessment to generate records.")
        else:
            history_df['Confidence (%)'] = (history_df['Confidence (%)'] * 100).round(2)
            history_df['Sex'] = history_df['sex'].map({1: 'Male', 0: 'Female'})
            history_df = history_df.drop('sex', axis=1)
            
            # Reorder columns slightly for better UI
            history_df = history_df[['timestamp', 'age', 'Sex', 'weight', 'BP', 'Cholesterol', 'Diagnosis', 'Confidence (%)']]
            
            st.dataframe(history_df, use_container_width=True)
            
            if len(history_df) > 1:
                st.markdown("### Recent Risk Trends")
                fig_trend = px.line(history_df, x='timestamp', y='Confidence (%)', color='Diagnosis', markers=True, title="Confidence of High-Risk Diagnoses Over Time")
                fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_trend, use_container_width=True)

    # ==========================================
    # PAGE 4: 📊 DATA EXPLORER
    # ==========================================
    elif menu_selection == "📊 Data Explorer":
        st.title("Clinical Data Explorer")
        st.markdown("Explore the underlying training dataset (`heart.csv`) to understand population trends.")
        st.markdown("---")
        
        try:
            df = pd.read_csv("data/heart.csv")
            st.dataframe(df, use_container_width=True, height=250)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Target Distribution")
                target_counts = df['target'].value_counts().reset_index()
                target_counts.columns = ['Status', 'Count']
                target_counts['Status'] = target_counts['Status'].map({
                    0: 'Healthy', 1: 'CAD', 2: 'Heart Attack', 3: 'Arrhythmia',
                    4: 'Heart Failure', 5: 'Valve Disease', 6: 'Cardiomyopathy',
                    7: 'Congenital', 8: 'Pericarditis', 9: 'Myocarditis',
                    10: 'Endocarditis', 11: 'Aortic Aneurysm', 12: 'PAD'
                })
                color_map = {
                    'Healthy': '#047857', 'CAD': '#e11d48', 'Heart Attack': '#7f1d1d',
                    'Arrhythmia': '#f59e0b', 'Heart Failure': '#dc2626',
                    'Valve Disease': '#7c3aed', 'Cardiomyopathy': '#db2777',
                    'Congenital': '#0891b2', 'Pericarditis': '#65a30d',
                    'Myocarditis': '#ea580c', 'Endocarditis': '#9333ea',
                    'Aortic Aneurysm': '#1d4ed8', 'PAD': '#b45309'
                }
                fig_pie = px.pie(target_counts, values='Count', names='Status',
                                 color='Status', color_discrete_map=color_map)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                st.subheader("Age vs Maximum Heart Rate")
                fig_scatter = px.scatter(df, x="age", y="thalach", color="target",
                                         color_continuous_scale=px.colors.sequential.RdBu_r)
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not load dataset: {e}")