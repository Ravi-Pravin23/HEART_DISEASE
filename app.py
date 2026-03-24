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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Force light color scheme for all browsers */
    :root {
        color-scheme: light !important;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #1e293b !important;
    }

    /* Soft cool gray/blue background matching reference */
    .stApp {
        background-color: #f4f7fb !important;
    }
    [data-testid="stMain"], [data-testid="block-container"] {
        background-color: #f4f7fb !important; 
        padding-top: 1rem;
    }

    /* Clean Poppins Headers */
    h1, h2, h3, h4, h5 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        color: #111827 !important;
        letter-spacing: -0.01em;
    }

    /* Custom Card Container - PCDP Style */
    .clinical-card {
        background-color: #ffffff;
        padding: 1.75rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: none;
        margin-bottom: 1.5rem;
        transition: box-shadow 0.2s ease-in-out;
    }
    .clinical-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }

    /* Sidebar: clean enterprise look */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] * {
        color: #4b5563 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* All text inputs: sharp borders, clear focus */
    input, textarea, [data-baseweb="input"] input,
    [data-baseweb="base-input"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.8rem !important;
        font-size: 0.95rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        box-shadow: none !important;
    }
    input:focus {
        border-color: #02aadb !important;
        box-shadow: 0 0 0 3px rgba(2, 170, 219, 0.15) !important;
    }

    /* Selectbox / dropdown */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    /* Labels */
    .stTextInput label, .stSelectbox label, .stSlider label,
    .stNumberInput label, label, p, span {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: transparent !important;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        color: #64748b !important;
        background-color: transparent !important;
        padding: 0.5rem 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        color: #02aadb !important;
        border-bottom: 2px solid #02aadb !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 600 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 500 !important;
        text-transform: none;
        font-size: 0.85rem !important;
    }

    /* Primary action buttons - PCDP Blue CTA */
    .stButton > button {
        background-color: #02aadb !important; 
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        font-size: 0.95rem;
        font-family: 'Poppins', sans-serif !important;
        transition: background-color 0.2s ease, transform 0.1s ease;
        box-shadow: 0 4px 6px rgba(2, 170, 219, 0.2);
    }
    .stButton > button:hover {
        background-color: #028eb8 !important; 
        color: white !important;
        transform: translateY(-1px); 
    }

    /* Download button - Dr.Doctor Orange/Red CTA */
    .stDownloadButton > button {
        background-color: #f05b41 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.95rem;
        box-shadow: 0 4px 6px rgba(240, 91, 65, 0.2);
    }
    .stDownloadButton > button:hover {
        background-color: #d14d35 !important;
        transform: translateY(-1px);
        color: white !important;
    }

    /* Expander header text */
    [data-testid="stExpander"] {
        border-radius: 10px !important;
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }
    [data-testid="stExpander"] summary span {
        color: #111827 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. AUTHENTICATION UI (Login / Register) ---
if not st.session_state['logged_in']:
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    _, auth_col, _ = st.columns([1, 1.2, 1])
    
    with auth_col:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #028eb8 0%, #02aadb 100%); padding: 2.5rem 1rem; text-align: center; color: white; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(2, 170, 219, 0.2);'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🩺</div>
                <h2 style='color: white !important; font-family: "Poppins", sans-serif; margin: 0; font-size: 1.8rem;'>Heart AI Portal</h2>
                <p style='color: rgba(255,255,255,0.9); font-size: 0.95rem; margin-top: 5px; font-weight: 400;'>Secure Access to Medical Records</p>
            </div>
        """, unsafe_allow_html=True)
        
        auth_tab1, auth_tab2 = st.tabs(["Sign In", "Register Provider"])
        
        with auth_tab1:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            log_user = st.text_input("Provider Email / ID", key="log_u", placeholder="dr.smith@clinic.com")
            log_pass = st.text_input("Password", type="password", key="log_p", placeholder="••••••••")
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            if st.button("Secure Login", use_container_width=True):
                users = load_users()
                if log_user in users and (users[log_user] == log_pass or users[log_user] == hash_password(log_pass)):
                    if users[log_user] == log_pass:
                        save_user(log_user, log_pass)
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = log_user
                    st.rerun()
                else:
                    st.error("Authentication failed.")
            
            st.markdown("""
                <div style='text-align: center; margin-top: 1.5rem;'>
                    <p style='font-size: 0.8rem; color: #64748b;'>By securely logging in, you agree to our <span style='color: #02aadb; cursor: pointer;'>Terms of Service</span></p>
                </div>
            """, unsafe_allow_html=True)

        with auth_tab2:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            new_user = st.text_input("Provider Email / ID", key="reg_u", placeholder="e.g. dr.smith@clinic.com")
            new_pass = st.text_input("Create Password", type="password", key="reg_p", placeholder="Minimum 8 characters")
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            if st.button("Create Provider Account", use_container_width=True):
                if new_user and new_pass:
                    save_user(new_user, new_pass)
                    st.success("Account provisioned successfully. Please sign in.")
                else:
                    st.warning("Please complete all fields.")
        
        st.markdown("""
            <div style='margin-top: 2rem; text-align: center; padding-top: 1rem; border-top: 1px solid #e2e8f0;'>
                <p style='margin: 0; font-size: 0.75rem; color: #94a3b8;'>© 2026 Heart AI Systems. All rights reserved.</p>
            </div>
        """, unsafe_allow_html=True)

# --- 5. MAIN APPLICATION (Visible after Login) ---
else:
    # --- Custom Top Navigation (Dr.Doctor Style) ---
    st.markdown("""
        <style>
        /* Hide the default sidebar completely */
        [data-testid="collapsedControl"] { display: none; }
        section[data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    
    # Top Bar Layout
    top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
    
    with top_col1:
        st.markdown(f"""
            <div style='display: flex; align-items: center; gap: 10px; padding-top: 5px;'>
                <span style='font-size: 2rem; color: #f05b41;'>⚕️</span>
                <span style='font-family: Poppins; font-weight: 700; font-size: 1.5rem; color: #02aadb;'>Heart AI</span>
            </div>
        """, unsafe_allow_html=True)
        
    with top_col2:
        # We use a horizontal radio button styled to look like top nav links
        menu_selection = st.radio(
            "Navigation",
            ["🩺 Assessment", "📈 Dashboard", "📂 Records", "⚙️ Settings"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
    with top_col3:
        # Right aligned logout action matching the Orange Signup button style
        st.markdown("<div style='text-align: right; padding-top: 5px;'>", unsafe_allow_html=True)
        if st.button("Logout Session", use_container_width=False, key="logout_btn"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.markdown(f"<span style='font-size: 0.9rem; color: #64748b; margin-right: 15px;'>Dr. {st.session_state['user_name']}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 2rem; border: none; border-bottom: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # --- SETTINGS / INTEGRATION PAGE ---
    if menu_selection == "⚙️ Settings":
        st.title("System Integration Settings")
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.subheader("n8n Automated Alerts")
        n8n_webhook_url = st.text_input(
            "Webhook Endpoint URL",
            value="http://localhost:5678/webhook/heart-alert",
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
        st.markdown("</div>", unsafe_allow_html=True)

    # PAGE 1: 🩺 DIAGNOSTIC ASSESSMENT
    elif menu_selection == "🩺 Assessment":
        st.title("Cardiovascular Assessment")
        
        # --- BATCH PROCESSING FEATURE ---
        with st.expander("📁 Batch Processing (CSV)", expanded=False):
            st.markdown("Upload multiple patient records for bulk AI stratification.")
            uploaded_file = st.file_uploader("", type="csv")
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(batch_df)} patient records.")
                    if st.button("Analyze Batch", use_container_width=True):
                        try:
                            model, scaler, features = load_models()
                            if not all(col in batch_df.columns for col in features):
                                st.warning(f"CSV missing columns: {features}")
                            else:
                                batch_input = batch_df[features].values
                                scaled_batch = scaler.transform(batch_input)
                                batch_preds = model.predict(scaled_batch)
                                batch_probs = np.max(model.predict_proba(scaled_batch), axis=1)
                                
                                disease_map = {0: 'Healthy', 1: 'CAD', 2: 'Heart Attack', 3: 'Arrhythmia', 4: 'Heart Failure', 5: 'Valve Disease', 6: 'Cardiomyopathy', 7: 'Congenital', 8: 'Pericarditis', 9: 'Myocarditis', 10: 'Endocarditis', 11: 'Aneurysm', 12: 'PAD'}
                                batch_df['Diagnosis'] = [disease_map.get(int(p), 'Unknown') for p in batch_preds]
                                batch_df['Confidence (%)'] = (batch_probs * 100).round(2)
                                
                                st.success("Batch analysis complete.")
                                st.dataframe(batch_df[['Diagnosis', 'Confidence (%)'] + features].head(10), use_container_width=True)
                                st.download_button("Export Batch Report", batch_df.to_csv(index=False).encode('utf-8'), "Batch_Report.csv", "text/csv")
                        except Exception as e: st.error(f"Error: {e}")
                except Exception as e: st.error(f"Read Error: {e}")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Build form in discrete cards
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0;'>Patient Demographics & Vitals</h3>", unsafe_allow_html=True)
        
        email_col, _ = st.columns([1.5, 1])
        with email_col:
            patient_email = st.text_input("Patient Notification Email", value="ravipravin2005@gmail.com", help="Report will be sent here.")
        
        st.markdown("<hr style='margin: 1rem 0; border: 0; border-top: 1px solid #e5e7eb;' />", unsafe_allow_html=True)
        
        col_demo, col_vitals = st.columns(2)
        with col_demo:
            age = st.number_input("Age (Years)", 1, 100, 45)
            sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            weight = st.number_input("Weight (kg)", 10, 250, 70)
            
        with col_vitals:
            trestbps = st.slider("Resting Systolic BP (mmHg)", 80, 250, 120)
            thalach = st.slider("Max Heart Rate (bpm)", 50, 250, 150)
            st.markdown("<p style='font-size: 0.8rem; color: #6b7280;'>Ensure patient was resting for 5 mins before BP reading.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0;'>Laboratory Findings & ECG</h3>", unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        with l1:
            chol = st.slider("Serum Chol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        with l2:
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
            exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        with l3:
            restecg = st.selectbox("ECG Pattern", [0, 1, 2], help="0: Normal, 1: ST-T abnormal, 2: LVH")
            oldpeak = st.slider("ST Depression", 0.0, 10.0, 1.0, 0.1)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0;'>Specialized Metrics</h3>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1: slope = st.selectbox("Peak ST Slope", [0, 1, 2])
        with s2: ca = st.selectbox("Fluoroscopy Vessels", [0, 1, 2, 3])
        with s3: thal = st.selectbox("Thal Status", [1, 2, 3])
        
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Clinical Profile", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if analyze_btn:
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

            # Update Session State for Persistence
            st.session_state['last_prediction'] = prediction
            st.session_state['last_prob'] = prob
            st.session_state['last_disease'] = disease_name
            st.session_state['last_vitals'] = (age, chol, trestbps, thalach, oldpeak, weight)

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
        
        # --- Results Display (Persistent) ---
        if 'last_prediction' in st.session_state:
            prediction = st.session_state['last_prediction']
            prob = st.session_state['last_prob']
            disease_name = st.session_state['last_disease']
            age, chol, trestbps, thalach, oldpeak, weight = st.session_state['last_vitals']

            st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top: 0; margin-bottom: 1.5rem;'>Diagnostic Summary & Analysis</h3>", unsafe_allow_html=True)
            
            # Vital Signs Metrics in a sleek row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Serum Cholesterol", f"{chol} mg/dl", delta="Elevated" if chol > 200 else "Optimal", delta_color="inverse" if chol > 200 else "normal")
            with m2:
                st.metric("Systemic Blood Pressure", f"{trestbps} mmHg", delta="Elevated" if trestbps > 130 else "Optimal", delta_color="inverse" if trestbps > 130 else "normal")
            with m3:
                st.metric("Max Heart Rate", f"{thalach} bpm")
            
            st.markdown("<hr style='margin: 1.5rem 0; border: 0; border-top: 1px solid #e5e7eb;' />", unsafe_allow_html=True)
            
            # Prediction Results & Gauge
            res_col, advice_col = st.columns([1, 1.2])
            with res_col:
                if prediction > 0:
                    st.error(f"### AI Finding: {disease_name}")
                else:
                    st.success(f"### AI Finding: {disease_name}")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    title = {'text': "Confidence Level", 'font': {'size': 14, 'color': '#6b7280'}},
                    number = {'suffix': "%", 'font': {'size': 36, 'color': '#111827', 'family': 'Outfit'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#2563eb" if prediction == 0 else "#ef4444"},
                        'bgcolor': "white",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': "#f3f4f6"},
                            {'range': [40, 75], 'color': "#e5e7eb"},
                            {'range': [75, 100], 'color': "#d1d5db"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with advice_col:
                st.markdown("<h4 style='color: #374151; margin-top: 0;'>Clinical Guidance Protocol</h4>", unsafe_allow_html=True)
                if prediction > 0:
                    st.warning(f"**Action Plan for {disease_name}**\n\nAutomated analysis indicates elevated risk factors. Consult the generated protocol PDF for detailed dietary, medicinal, and clinical follow-up guidelines specific to this risk profile.")
                else:
                    st.info(f"**Action Plan for {disease_name}**\n\nAutomated analysis indicates baseline health metrics. Standard cardiovascular preventative measures are recommended.")
                
            st.markdown("</div>", unsafe_allow_html=True)

            # --- Radar & Factors in two clean cards ---
            c_radar, c_factors = st.columns(2)
            with c_radar:
                st.markdown("<div class='clinical-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top: 0;'>Biomarker Profile Visualization</h4>", unsafe_allow_html=True)
                categories = ['Age', 'BP', 'Chol', 'HR', 'ST Dep', 'Weight']
                patient_values = [min(100, (age/100)*100), min(100, (trestbps/200)*100), min(100, (chol/400)*100), min(100, (thalach/200)*100), min(100, (oldpeak/6)*100), min(100, (weight/150)*100)]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=patient_values, 
                    theta=categories, 
                    fill='toself', 
                    name='Patient', 
                    line_color='#2563eb', 
                    fillcolor='rgba(37, 99, 235, 0.15)'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=False, range=[0, 100]),
                        angularaxis=dict(gridcolor='#e5e7eb', linecolor='#e5e7eb')
                    ), 
                    height=280, 
                    margin=dict(l=20, r=20, t=20, b=20), 
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with c_factors:
                st.markdown("<div class='clinical-card' style='height: 100%;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='margin-top: 0;'>Key Predictive Drivers</h4>", unsafe_allow_html=True)
                model, _, features = load_models()
                importances = model.feature_importances_
                feat_importances = pd.Series(importances, index=features).nlargest(5)
                fig_bar = px.bar(
                    x=feat_importances.values, 
                    y=feat_importances.index, 
                    orientation='h', 
                    color_discrete_sequence=['#2563eb']
                )
                fig_bar.update_layout(
                    height=280, 
                    margin=dict(l=20, r=20, t=20, b=20), 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    yaxis={'categoryorder':'total ascending'},
                    xaxis_title="Influence Score",
                    yaxis_title=""
                )
                fig_bar.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb', zeroline=False)
                fig_bar.update_yaxes(showgrid=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            pdf_bytes = create_pdf(st.session_state['user_name'], prediction, prob, age, chol, trestbps, weight)
            st.download_button(
                label="Generate Official Medical Report (PDF)",
                data=pdf_bytes,
                file_name=f"Clinical_Report_{st.session_state['user_name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # ==========================================
    # PAGE 2: 📈 CLINIC DASHBOARD
    # ==========================================
    elif menu_selection == "📈 Patient Dashboard":
        st.title("Clinic Insights Dashboard")
        st.markdown("<p style='color:#6b7280; font-size: 1.1rem; margin-top:-1rem;'>Real-time clinical metrics and screening volume.</p>", unsafe_allow_html=True)
        
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top: 0; margin-bottom: 1rem;'>Practice KPI Overview</h4>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Screenings", "1,204", "+15 this week")
        c2.metric("High-Risk Cases Identified", "241", "19.8% detection rate")
        c3.metric("Avg. Assessment Time", "4.2m", "-30s improvement")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top: 0;'>Monthly Screening Trends</h4>", unsafe_allow_html=True)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        screenings = [120, 150, 180, 170, 210, 190]
        fig_line = px.line(x=months, y=screenings, markers=True, color_discrete_sequence=['#2563eb'])
        fig_line.update_layout(
            height=350, 
            margin=dict(l=20, r=20, t=20, b=20), 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Month",
            yaxis_title="Total Assessments"
        )
        fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
        fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb', zeroline=False)
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # PAGE 3: 📂 PATIENT HISTORY
    # ==========================================
    elif menu_selection == "📂 Patient History":
        st.title("Digital Health Records")
        st.markdown(f"<p style='color:#6b7280; font-size: 1.1rem; margin-top:-1rem;'>Secure access to diagnostic history for Dr. {st.session_state['user_name']}</p>", unsafe_allow_html=True)
        conn = sqlite3.connect(PATIENTS_DB)
        history_df = pd.read_sql_query("SELECT timestamp, age, sex, weight, trestbps, chol, prediction_str, probability FROM patients WHERE doctor_name = ? ORDER BY timestamp DESC", conn, params=(st.session_state['user_name'],))
        conn.close()
        
        if history_df.empty:
            st.info("No clinical records found for this specialist session.")
        else:
            st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
            history_df['Confidence'] = (history_df['probability'] * 100).round(1).astype(str) + "%"
            history_df['Sex'] = history_df['sex'].map({1: 'M', 0: 'F'})
            display_df = history_df[['timestamp', 'age', 'Sex', 'weight', 'prediction_str', 'Confidence']]
            st.dataframe(display_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # PAGE 4: 📊 DATA EXPLORER
    # ==========================================
    elif menu_selection == "📊 Data Explorer":
        st.title("Clinical Data Explorer")
        st.markdown("<p style='color:#64748b; margin-top:-1rem;'>Investigating the underlying heart disease dataset.</p>", unsafe_allow_html=True)
        
        try:
            df = pd.read_csv("data/heart.csv")
            st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
            st.dataframe(df.head(100), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
                st.subheader("condition Distribution")
                target_counts = df['target'].value_counts().reset_index()
                target_counts.columns = ['Status', 'Count']
                fig_pie = px.pie(target_counts, values='Count', names='Status', hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe)
                fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
                st.subheader("Age vs Max Heart Rate")
                fig_scatter = px.scatter(df, x="age", y="thalach", color="target", color_continuous_scale='RdBu')
                fig_scatter.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Explorer Error: {e}")