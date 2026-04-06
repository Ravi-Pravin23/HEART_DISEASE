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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_resource
def load_models():
    """Loads the AI models and scalers securely."""
    return joblib.load('models/heart_model.pkl'), joblib.load('models/scaler.pkl'), joblib.load('models/features.pkl')

def create_pdf(patient_name, doctor_name, pred, probability, age_val, chol_val, bp_val, weight_val):
    """Generates a downloadable PDF report for the patient."""
    from datetime import datetime
    pdf = FPDF()
    pdf.add_page()
    
    # Header styling
    pdf.set_font("Helvetica", size=20, style="B")
    pdf.set_text_color(2, 170, 219) # Primary brand color
    pdf.cell(0, 10, txt="Heart AI Clinical Portal", ln=True, align='L')
    
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, txt="123 Advanced Cardiology Way, Metro Health District", ln=True, align='L')
    pdf.cell(0, 5, txt="Email: care@heartai.clinic | Phone: (555) 019-2038", ln=True, align='L')
    
    # Line break
    pdf.ln(5)
    pdf.set_draw_color(220, 220, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Title
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, txt="OFFICIAL DIAGNOSTIC ASSESSMENT REPORT", ln=True, align='C')
    pdf.ln(5)
    
    # Metadata
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(40, 8, txt="Date of Assessment:", border=0)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(60, 8, txt=datetime.now().strftime("%B %d, %Y - %H:%M"), border=0)
    
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(40, 8, txt="Attending Physician:", border=0)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, txt=f"Dr. {doctor_name}", border=0, ln=True)
    
    pdf.ln(5)
    
    # Patient Demographics block
    pdf.set_fill_color(245, 247, 250)
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(0, 8, txt=" 1. PATIENT DEMOGRAPHICS & VITALS", border=1, ln=True, fill=True)
    
    pdf.set_font("Helvetica", size=10)
    pdf.cell(40, 8, txt="Patient Name:", border="L")
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(60, 8, txt=f"{patient_name.strip() if patient_name else 'Anonymous Subject'}", border=0)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(40, 8, txt="Age (Years):", border=0)
    pdf.cell(0, 8, txt=f"{age_val}", border="R", ln=True)
    
    pdf.cell(40, 8, txt="Weight (kg):", border="L")
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(60, 8, txt=f"{weight_val}", border=0)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(40, 8, txt="Blood Pressure:", border=0)
    pdf.cell(0, 8, txt=f"{bp_val} mmHg", border="R", ln=True)

    pdf.cell(40, 8, txt="Total Cholesterol:", border="L, B")
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(60, 8, txt=f"{chol_val} mg/dl", border="B")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(40, 8, txt="", border="B")
    pdf.cell(0, 8, txt="", border="R, B", ln=True)
    
    pdf.ln(10)
    
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
    
    # Assessment block
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(0, 8, txt=" 2. AI DIAGNOSTIC FINDINGS", border=1, ln=True, fill=True)
    
    pdf.set_font("Helvetica", size=10)
    pdf.cell(50, 10, txt="Primary Diagnosis:", border="L")
    pdf.set_font("Helvetica", size=11, style="B")
    if pred > 0:
        pdf.set_text_color(225, 29, 72) # Red for risk
    else:
        pdf.set_text_color(5, 150, 105) # Green for healthy
    pdf.cell(0, 10, txt=f"{status}", border="R", ln=True)
    
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(50, 10, txt="Model Confidence Level:", border="L, B")
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(0, 10, txt=f"{probability*100:.2f}%", border="R, B", ln=True)
    
    pdf.ln(10)
    
    # Clinical Remarks
    pdf.set_font("Helvetica", size=10, style="B")
    pdf.cell(0, 8, txt=" 3. CLINICAL REMARKS & RECOMMENDATIONS", border=1, ln=True, fill=True)
    pdf.set_font("Helvetica", size=10)
    if pred > 0:
        recommendation = ("This AI screening has identified elevated risk factors associated "
                          f"with {status}. It is strongly recommended that the patient undergoes "
                          "statutory diagnostic verifications, including an echocardiogram and a complete "
                          "metabolic panel. Ensure strict monitoring of blood pressure and cholesterol.")
    else:
        recommendation = ("This AI screening does not detect immediate cardiovascular abnormalities. "
                          "The patient presents a low-risk profile. Routine health maintenance, including "
                          "balanced diet and periodic checkups, is recommended to preserve baseline vitals.")
    pdf.multi_cell(0, 6, txt=recommendation, border="L, R, B", align="L")
    
    pdf.ln(15)
    
    # Physician signature
    pdf.cell(100, 10, txt="", border=0)
    pdf.cell(90, 10, txt="Electronically Signed By:", border=0, ln=True, align="C")
    pdf.set_font("Helvetica", size=12, style="I")
    pdf.cell(100, 10, txt="", border=0)
    pdf.cell(90, 10, txt=f"Dr. {doctor_name}", border=0, ln=True, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(100, 10, txt="", border=0)
    pdf.set_draw_color(100, 100, 100)
    pdf.line(120, pdf.get_y(), 190, pdf.get_y())
    pdf.cell(90, 10, txt="Department of Cardiology", border=0, ln=True, align="C")
    
    # Footer disclaimer
    pdf.set_y(-30)
    pdf.set_font("Helvetica", size=8, style="I")
    pdf.set_text_color(150, 150, 150)
    disclaimer = ("DISCLAIMER: This document is an AI-assisted preliminary medical report and does not constitute "
                  "a definitive clinical diagnosis. Final medical decisions must be carried out by certified healthcare "
                  "professionals. Protected Health Information (PHI) contained in this report is strictly confidential.")
    pdf.multi_cell(0, 4, txt=disclaimer, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

def send_smtp_email(to_email, subject, body, attachment_bytes=None, attachment_name="Clinical_Report.pdf"):
    """Sends a direct SMTP email (e.g., via Gmail)."""
    smtp_server = st.session_state.get('smtp_server', "smtp.gmail.com")
    smtp_port = st.session_state.get('smtp_port', 587)
    smtp_user = st.session_state.get('smtp_user', "")
    smtp_pass = st.session_state.get('smtp_pass', "")

    if not smtp_user or not smtp_pass:
        return False, "SMTP credentials not configured in Settings."

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        if attachment_bytes:
            part = MIMEApplication(attachment_bytes, Name=attachment_name)
            part['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
            msg.attach(part)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# --- 1. DATABASE SETUP (FR-1: User Registration) ---
def get_db_connection():
    """Returns a PostgreSQL connection if secrets are configured, else SQLite."""
    err_msg = ""
    try:
        # Check if secrets contain the postgresql URL
        if "connections" in st.secrets and "postgresql" in st.secrets["connections"]:
            # Test the connection to ensure it works
            conn = st.connection("postgresql", type="sql")
            return conn, "postgresql", ""
    except Exception as e:
        err_msg = str(e)
    
    # Fallback to SQLite
    return sqlite3.connect("data/patients.db", check_same_thread=False), "sqlite", err_msg

def init_db():
    conn, mode, _ = get_db_connection()
    if mode == "postgresql":
        with conn.session as s:
            s.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                full_name TEXT
            )""")
            s.commit()
    else:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
        try:
            c.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()

def init_patients_db():
    conn, mode, _ = get_db_connection()
    if mode == "postgresql":
        with conn.session as s:
            s.execute("""CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                doctor_name TEXT,
                patient_name TEXT,
                age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER, chol INTEGER,
                fbs INTEGER, restecg INTEGER, thalach INTEGER, exang INTEGER,
                oldpeak REAL, slope INTEGER, ca INTEGER, thal INTEGER, weight INTEGER,
                prediction_str TEXT, probability REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
            s.commit()
    else:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_name TEXT,
            age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER, chol INTEGER,
            fbs INTEGER, restecg INTEGER, thalach INTEGER, exang INTEGER,
            oldpeak REAL, slope INTEGER, ca INTEGER, thal INTEGER, weight INTEGER,
            prediction_str TEXT, probability REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        try: c.execute("ALTER TABLE patients ADD COLUMN weight INTEGER DEFAULT 70")
        except: pass
            
        try: c.execute("ALTER TABLE patients ADD COLUMN patient_name TEXT")
        except: pass
        
        conn.commit()
        conn.close()

def save_patient_record(doctor, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, weight, pred_str, prob):
    conn, mode, _ = get_db_connection()
    if mode == "postgresql":
        sql = """INSERT INTO patients (
            doctor_name, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal, weight, prediction_str, probability
        ) VALUES (:doctor_name, :patient_name, :age, :sex, :cp, :trestbps, :chol, :fbs, :restecg, 
                  :thalach, :exang, :oldpeak, :slope, :ca, :thal, :weight, :prediction_str, :probability)"""
        values = {
            "doctor_name": doctor, "patient_name": patient_name, "age": age, "sex": sex, "cp": cp, 
            "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach, 
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, 
            "weight": weight, "prediction_str": pred_str, "probability": prob
        }
        with conn.session as s:
            s.execute(sql, values)
            s.commit()
    else:
        sql = '''INSERT INTO patients (
            doctor_name, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal, weight, prediction_str, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        c = conn.cursor()
        c.execute(sql, (doctor, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, weight, pred_str, prob))
        conn.commit()
        conn.close()

def load_users():
    conn, mode, _ = get_db_connection()
    if mode == "postgresql":
        with conn.session as s:
            res = s.execute("SELECT username, password, full_name FROM users").fetchall()
            users = {row[0]: row[1] for row in res}
            full_names = {row[0]: (row[2] if row[2] else row[0]) for row in res}
            return users, full_names
    else:
        c = conn.cursor()
        c.execute("SELECT username, password FROM users")
        users = {row[0]: row[1] for row in c.fetchall()}
        
        full_names = {}
        try:
            c.execute("SELECT username, full_name FROM users")
            full_names = {row[0]: (row[1] if row[1] else row[0]) for row in c.fetchall()}
        except sqlite3.OperationalError:
            full_names = {k: k for k in users.keys()}
        conn.close()
        return users, full_names

def save_user(username, password, full_name=None):
    conn, mode, _ = get_db_connection()
    hashed_pwd = hash_password(password)
    
    if mode == "postgresql":
        sql = """INSERT INTO users (username, password, full_name) VALUES (:u, :p, :f)
                 ON CONFLICT (username) DO UPDATE SET password = :p, full_name = :f"""
        with conn.session as s:
            s.execute(sql, {"u": username, "p": hashed_pwd, "f": full_name})
            s.commit()
    else:
        c = conn.cursor()
        c.execute("REPLACE INTO users (username, password, full_name) VALUES (?, ?, ?)", (username, hashed_pwd, full_name))
        conn.commit()
        conn.close()

def migrate_json_to_sqlite():
    if os.path.exists("data/users.json"):
        with open("data/users.json", "r") as f:
            try:
                old_users = json.load(f)
                conn = sqlite3.connect("data/patients.db")
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
if 'n8n_webhook_url' not in st.session_state:
    st.session_state['n8n_webhook_url'] = "http://localhost:5678/webhook/heart-alert"
if 'email_provider' not in st.session_state:
    st.session_state['email_provider'] = "SMTP (Direct)"
if 'smtp_server' not in st.session_state:
    st.session_state['smtp_server'] = "smtp.gmail.com"
if 'smtp_port' not in st.session_state:
    st.session_state['smtp_port'] = 587
if 'smtp_user' not in st.session_state:
    st.session_state['smtp_user'] = ""
if 'smtp_pass' not in st.session_state:
    st.session_state['smtp_pass'] = ""

# --- 3. UI CONFIGURATION ---
st.set_page_config(
    page_title="Heart AI Clinical Portal",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" rel="stylesheet">
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
    /* Keep Material icon ligatures rendering as icons in the sidebar toggle */
    section[data-testid="stSidebar"] span[class*="material-symbols"],
    section[data-testid="stSidebar"] i[class*="material-icons"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    }
    /* Material Symbols default rendering (ligatures) */
    .material-symbols-rounded {
        font-variation-settings: 'FILL' 0, 'wght' 500, 'GRAD' 0, 'opsz' 24;
        line-height: 1;
        vertical-align: middle;
    }

    /* Fallback: if icon font fails, show arrows instead of ligature text */
    /* Fallback: force a single clean double-arrow glyph */
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarExpandButton"] span {
        display: none !important; /* hide ligature text + any nested spans */
    }
    /* Center the glyph in the sidebar toggle */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarExpandButton"] {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 42px !important;     /* make it easy to touch */
        height: 42px !important;    /* make it easy to touch */
        min-width: 42px !important;
        min-height: 42px !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.95) !important;
        border: 1px solid #e5e7eb !important;
        cursor: pointer !important;
        padding: 0 !important;
        z-index: 50 !important;
    }
    [data-testid="stSidebarCollapseButton"]::before {
        content: "«" !important;
        font-size: 20px !important;
        color: #334155 !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        line-height: 1 !important;
        pointer-events: none !important; /* ensure click works on the button */
    }
    [data-testid="stSidebarExpandButton"]::before {
        content: "»" !important;
        font-size: 20px !important;
        color: #334155 !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        line-height: 1 !important;
        pointer-events: none !important; /* ensure click works on the button */
    }
    [data-testid="stSidebarCollapseButton"]:hover::before,
    [data-testid="stSidebarExpandButton"]:hover::before {
        color: #0f172a !important;
    }

    /* Sidebar navigation (radio) polish */
    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.75rem !important;
        margin: 0.25rem 0 !important;
        transition: all 0.15s ease-in-out;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: #02aadb !important;
        box-shadow: 0 0 0 3px rgba(2, 170, 219, 0.10) !important;
        transform: translateY(-1px);
    }
    /* Selected option */
    section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    /* Tighten spacing around sidebar widgets */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    /* Input Container Blocks (PCDP Style Solid Fill) */
    /* Input Container Blocks (PCDP Style Solid Fill) - Fix for eye icon double border */
    .stTextInput > div > div, .stSelectbox > div > div {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
        overflow: hidden;
    }
    
    .stTextInput > div > div:focus-within, .stSelectbox > div > div:focus-within {
        background-color: #ffffff !important;
        border-color: #8b5cf6 !important; /* PCDP Purple */
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }

    /* Override base-web defaults leaking through */
    [data-baseweb="base-input"], [data-baseweb="input"], [data-baseweb="select"] > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Inner Input Elements */
    input, textarea {
        background-color: transparent !important;
        border: none !important;
        color: #334155 !important;
        font-size: 1rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        box-shadow: none !important;
        padding: 0.6rem 0.5rem !important;
    }
    input[type="password"] {
        padding-right: 3rem !important;
    }
    input:focus, textarea:focus {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background-color: transparent !important;
    }
    
    /* Hide the browser's default password reveal eye icon to prevent double icons */
    input::-ms-reveal,
    input::-ms-clear {
        display: none !important;
    }
    input::-webkit-contacts-auto-fill-button, 
    input::-webkit-credentials-auto-fill-button {
        visibility: hidden !important;
        pointer-events: none !important;
        position: absolute !important;
        right: 0 !important;
    }



    /* Labels */
    .stTextInput label, .stSelectbox label, .stSlider label,
    .stNumberInput label, label, p {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Hide 'Press Enter to apply' instruction to fix overlapping */
    [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Do not override Streamlit icon ligatures (expander/toggle chevrons) */
    span[class*="material-symbols"],
    i[class*="material-icons"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
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
    
    /* Sidebar specific button styles to prevent text wrapping */
    section[data-testid="stSidebar"] .stButton > button {
        padding: 0.5rem 0.25rem;
        font-size: 0.85rem;
        white-space: nowrap;
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
    st.markdown("""
        <style>
        /* Stronger targeting for the Auth Column to ensure the white card is visible */
        [data-testid="column"]:nth-of-type(2), [data-testid="stColumn"]:nth-of-type(2) {
            background-color: #ffffff !important;
            padding: 3rem 2.5rem !important;
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1) !important;
        }
        .auth-btn > button {
            background-color: #8b5cf6 !important;
            border: none !important;
            box-shadow: 0 4px 10px rgba(139, 92, 246, 0.2) !important;
            margin-top: 10px;
        }
        .auth-btn > button:hover {
            background-color: #7c3aed !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: none !important;
            justify-content: center;
            margin-bottom: 0.5rem;
        }
        .stTabs [aria-selected="true"] {
            color: #8b5cf6 !important;
            border-bottom: 2px solid #8b5cf6 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    _, auth_col, _ = st.columns([1, 1.2, 1])
    
    with auth_col:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <div style='display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 5px;'>
                    <span style='font-size: 2.2rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));'>❤️</span>
                    <span style='font-family: Poppins; font-weight: 600; font-size: 1.5rem; color: #0f172a;'>Heart AI Portal</span>
                </div>
                <h3 style='color: #8b5cf6 !important; font-family: "Poppins", sans-serif; margin: 0; font-size: 1rem; font-weight: 500;'>Empowering Urban Cardiology with Precision AI</h3>
            </div>
        """, unsafe_allow_html=True)
        
        auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])
        
        with auth_tab1:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            log_user = st.text_input("Username", key="log_u")
            log_pass = st.text_input("Password", type="password", key="log_p")
            
            st.markdown("<div class='auth-btn'>", unsafe_allow_html=True)
            if st.button("Login", width="stretch"):
                users, full_names = load_users()
                if log_user in users and (users[log_user] == log_pass or users[log_user] == hash_password(log_pass)):
                    if users[log_user] == log_pass:
                        save_user(log_user, log_pass, full_names.get(log_user))
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = log_user
                    st.session_state['full_name'] = full_names.get(log_user, log_user)
                    st.rerun()
                else:
                    st.error("Authentication failed.")
            st.markdown("</div>", unsafe_allow_html=True)
        with auth_tab2:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            
            new_name = st.text_input("Full Name", key="reg_name", placeholder="e.g., Dr. Ravi Pravin")
            new_email = st.text_input("Email Address", key="reg_email", placeholder="e.g., ravipravin@clinic.com")
            new_user = st.text_input("Username", key="reg_u")
            new_pass = st.text_input("Create Password", type="password", key="reg_p")
            confirm_pass = st.text_input("Confirm Password", type="password", key="reg_p_conf")
            
            st.markdown("<div class='auth-btn'>", unsafe_allow_html=True)
            if st.button("Register", width="stretch"):
                if new_name and new_email and new_user and new_pass and confirm_pass:
                    if new_pass == confirm_pass:
                        save_user(new_user, new_pass, new_name)
                        st.success("Account provisioned successfully. Please switch to Login.")
                    else:
                        st.error("Passwords do not match. Please try again.")
                else:
                    st.warning("Please complete all required fields.")
            st.markdown("</div>", unsafe_allow_html=True)
            
# --- 5. MAIN APPLICATION (Visible after Login) ---
else:
    # --- Vertical Sidebar Navigation ---
    with st.sidebar:
        st.markdown(f"""
            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 2rem;'>
                <span style='font-size: 2rem; color: #f05b41;'>⚕️</span>
                <span style='font-family: Poppins; font-weight: 700; font-size: 1.5rem; color: #02aadb;'>Heart AI</span>
            </div>
        """, unsafe_allow_html=True)

        def _set_page(page: str):
            st.session_state["active_page"] = page

        if "active_page" not in st.session_state:
            st.session_state["active_page"] = "🩺 Assessment"

        # st.caption("Use the top-left arrow to expand/collapse the menu.")
        st.markdown(
            "<div style='font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; color: #94a3b8; margin: 0.25rem 0;'>MENU</div>",
            unsafe_allow_html=True,
        )

        nav_query = st.text_input("Search menu", placeholder="Type to filter…", label_visibility="collapsed")

        PAGES = [
            ("Clinical", "🩺 Assessment"),
            ("Clinical", "📂 Records"),
            ("Insights", "📈 Dashboard"),
            ("Insights", "📊 Data Explorer"),
        ]

        def _matches(q: str, label: str) -> bool:
            q = (q or "").strip().lower()
            if not q:
                return True
            return q in label.lower()

        # Sectioned, searchable navigation (buttons)
        current_section = None
        for section, label in PAGES:
            if not _matches(nav_query, label):
                continue
            if section != current_section:
                if current_section is not None:
                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; color: #cbd5e1; margin: 0.25rem 0;'>{section.upper()}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<hr style='margin: 0.25rem 0 0.5rem 0; border: 0; border-top: 1px solid #e2e8f0;' />", unsafe_allow_html=True)
                current_section = section

            is_active = st.session_state["active_page"] == label
            btn_label = f"• {label}" if is_active else label
            if st.button(btn_label, width="stretch", key=f"nav_{label}"):
                _set_page(label)

        # Quick actions
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; color: #cbd5e1; margin: 0.25rem 0;'>QUICK ACTIONS</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin: 0.25rem 0 0.5rem 0; border: 0; border-top: 1px solid #e2e8f0;' />", unsafe_allow_html=True)

        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("New assessment", width="stretch", key="qa_new_assessment"):
                st.session_state.pop("last_prediction", None)
                _set_page("🩺 Assessment")
        with qa2:
            if st.button("Export records", width="stretch", key="qa_export_records"):
                st.session_state["export_records_hint"] = True
                _set_page("📂 Records")

        # n8n test button removed per user request
        
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        display_name = st.session_state.get('full_name', st.session_state.get('user_name', ''))
        initials = "".join([p[0].upper() for p in str(display_name).split()[:2]]) or "DR"
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; padding:10px 12px; border:1px solid #e2e8f0; border-radius:12px; background:#ffffff;">
              <div style="width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; background:#e0f2fe; color:#0284c7; font-weight:700;">
                {initials}
              </div>
              <div style="line-height:1.1;">
                <div style="font-weight:700; color:#0f172a; font-size:0.95rem;">Dr. {display_name}</div>
                <div style="color:#94a3b8; font-size:0.8rem;">Clinician</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Logout", width="stretch", key="logout_btn"):
            st.session_state['logged_in'] = False
            st.rerun()

        # --- Database Status Indicator ---
        st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: 0; border-top: 1px dashed #e2e8f0;' />", unsafe_allow_html=True)
        conn, mode, err = get_db_connection()
        if mode == "postgresql":
            st.markdown("""
                <div style='display:flex; align-items:center; gap:8px; background:#f0fdf4; border-radius:10px; padding:10px; border:1px solid #bbf7d0;'>
                    <span style='color:#16a34a; font-size:1.4rem;'>●</span>
                    <div style='display:flex; flex-direction:column; line-height:1.2;'>
                        <span style='font-size:0.65rem; color:#16a34a; font-weight:800; text-transform:uppercase; letter-spacing:0.05em;'>Active Connection</span>
                        <span style='font-size:0.85rem; color:#166534; font-weight:700;'>🌐 Cloud Database</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            status_text = "Secrets Missing" if "connections" not in st.secrets else "Connection Error"
            st.markdown(f"""
                <div style='display:flex; align-items:center; gap:8px; background:#fef2f2; border-radius:10px; padding:10px; border:1px solid #fecaca;'>
                    <span style='color:#dc2626; font-size:1.4rem;'>●</span>
                    <div style='display:flex; flex-direction:column; line-height:1.2;'>
                        <span style='font-size:0.65rem; color:#dc2626; font-weight:800; text-transform:uppercase; letter-spacing:0.05em;'>{status_text}</span>
                        <span style='font-size:0.85rem; color:#991b1b; font-weight:700;'>📁 Local SQLite</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if err:
                with st.expander("Show connection error"):
                    st.code(err, language="text")

    menu_selection = st.session_state["active_page"]

    # PAGE 1: 🩺 DIAGNOSTIC ASSESSMENT
    if menu_selection == "🩺 Assessment":
        st.title("Cardiovascular Assessment")
        
        # --- BATCH PROCESSING FEATURE ---
        with st.expander("📁 Batch Processing (CSV)", expanded=False):
            st.markdown("Upload multiple patient records for bulk AI stratification.")
            uploaded_file = st.file_uploader("Upload CSV file", type="csv", label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(batch_df)} patient records.")
                    if st.button("Analyze Batch", width="stretch"):
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
                                st.dataframe(batch_df[['Diagnosis', 'Confidence (%)'] + features].head(10), width="stretch")
                                st.download_button("Export Batch Report", batch_df.to_csv(index=False).encode('utf-8'), "Batch_Report.csv", "text/csv")
                        except Exception as e: st.error(f"Error: {e}")
                except Exception as e: st.error(f"Read Error: {e}")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Build form in discrete cards
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0;'>Patient Demographics & Vitals</h3>", unsafe_allow_html=True)
        
        patient_name = st.text_input("Patient Full Name", placeholder="e.g., John Doe")
        
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
        analyze_btn = st.button("Analyze Clinical Profile", width="stretch")
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
            st.session_state['last_patient_name'] = patient_name

            # --- Save to Patient History Database ---
            try:
                save_patient_record(
                    st.session_state['user_name'], patient_name, age, sex, cp, trestbps, chol, fbs, restecg, 
                    thalach, exang, oldpeak, slope, ca, thal, weight, disease_name, prob
                )
                st.toast(f"Record for {patient_name} saved to secure clinical vault.", icon="✅")
            except Exception as e:
                st.error(f"Clinical Database Error: {e}")


        
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
                st.plotly_chart(fig_gauge, width="stretch")

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
                st.plotly_chart(fig_radar, width="stretch")
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
                st.plotly_chart(fig_bar, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            pdf_bytes = create_pdf(st.session_state.get('last_patient_name', ''), st.session_state['user_name'], prediction, prob, age, chol, trestbps, weight)
            
            # format patient name for filename or fallback
            p_name_file = st.session_state.get('last_patient_name', 'Patient')
            if not p_name_file.strip():
                p_name_file = "Patient"
                
            st.download_button(
                label="Generate Official Medical Report (PDF)",
                data=pdf_bytes,
                file_name=f"Clinical_Report_{p_name_file.replace(' ', '_')}.pdf",
                mime="application/pdf",
                width="stretch"
            )

    # ==========================================
    # PAGE 2: 📈 CLINIC DASHBOARD
    # ==========================================
    elif menu_selection == "📈 Dashboard":
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
        st.plotly_chart(fig_line, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # PAGE 3: 📂 PATIENT HISTORY
    # ==========================================
    elif menu_selection == "📂 Records":
        st.title("Digital Health Records")
        st.markdown(f"<p style='color:#6b7280; font-size: 1.1rem; margin-top:-1rem;'>Secure access to diagnostic history for Dr. {display_name}</p>", unsafe_allow_html=True)
        
        conn, mode, _ = get_db_connection()
        query = "SELECT id, timestamp, patient_name, age, sex, weight, trestbps, chol, prediction_str, probability FROM patients WHERE doctor_name = :d ORDER BY timestamp DESC"
        
        if mode == "postgresql":
            # with conn.session as s:
            #     history_df = pd.read_sql_query(query, s.connection(), params={"d": st.session_state['user_name']})
            history_df = conn.query(query.replace(":d", f"'{st.session_state['user_name']}'"), ttl=0)
        else:
            history_df = pd.read_sql_query(query.replace(":d", "?"), conn, params=(st.session_state['user_name'],))
            conn.close()
        
        if history_df.empty:
            st.info("No clinical records found for this specialist session.")
        else:
            st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
            history_df['Confidence'] = (history_df['probability'] * 100).round(1).astype(str) + "%"
            history_df['Sex'] = history_df['sex'].map({1: 'M', 0: 'F'})
            
            if 'editor_key_counter' not in st.session_state:
                st.session_state['editor_key_counter'] = 0
            if 'select_all_state' not in st.session_state:
                st.session_state['select_all_state'] = False

            st.markdown("<p style='font-size: 0.9rem; color: #64748b; margin-bottom: 0.2rem;'>Search, select, and manage your clinical records below.</p>", unsafe_allow_html=True)
            
            search_query = st.text_input("🔍 Search", placeholder="Type name, diagnosis, etc. to filter...", label_visibility="collapsed")
            if search_query:
                # Filter history_df based on search_query matching any pertinent column
                searchable_cols = ['timestamp', 'patient_name', 'age', 'Sex', 'weight', 'prediction_str']
                mask = history_df[searchable_cols].astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
                history_df = history_df[mask]
                
            if history_df.empty:
                st.warning("No records match your search query.")
            else:
                col1, col2, _ = st.columns([1, 1, 4])
                with col1:
                    if st.button("☑️ Select All", use_container_width=True):
                        st.session_state['select_all_state'] = True
                        st.session_state['editor_key_counter'] += 1
                        st.rerun()
                with col2:
                    if st.button("☐ Clear All", use_container_width=True):
                        st.session_state['select_all_state'] = False
                        st.session_state['editor_key_counter'] += 1
                        st.rerun()
                
                # Add a selection column for deletion
                history_df.insert(0, "Select", st.session_state['select_all_state'])
            
                # We use st.data_editor to allow row selection
                display_cols = ['Select', 'timestamp', 'patient_name', 'age', 'Sex', 'weight', 'prediction_str', 'Confidence']
                
                edited_df = st.data_editor(
                    history_df[display_cols],
                    hide_index=True,
                    use_container_width=True,
                    disabled=['timestamp', 'patient_name', 'age', 'Sex', 'weight', 'prediction_str', 'Confidence'],
                    column_config={"Select": st.column_config.CheckboxColumn("Select", default=False)},
                    key=f"record_editor_{st.session_state['editor_key_counter']}"
                )
                
                # Find selected IDs
                selected_labels = edited_df.index[edited_df['Select']].tolist()
                selected_ids = history_df.loc[selected_labels]['id'].tolist()
                
                if selected_ids:
                    del_col, exp_col = st.columns(2)
                    with del_col:
                        if st.button(f"🗑️ Delete Selected ({len(selected_ids)})", type="primary", use_container_width=True):
                            conn, mode, _ = get_db_connection()
                            if mode == "postgresql":
                                sql = f"DELETE FROM patients WHERE id IN ({','.join([str(i) for i in selected_ids])})"
                                with conn.session as s:
                                    s.execute(sql)
                                    s.commit()
                            else:
                                c = conn.cursor()
                                c.execute(f"DELETE FROM patients WHERE id IN ({','.join(['?']*len(selected_ids))})", selected_ids)
                                conn.commit()
                                conn.close()
                            st.session_state['select_all_state'] = False
                            st.session_state['editor_key_counter'] += 1
                            st.success("Selected patient records deleted securely.")
                            st.rerun()
                    with exp_col:
                        export_selected_df = history_df.loc[selected_labels][['timestamp', 'patient_name', 'age', 'Sex', 'weight', 'prediction_str', 'Confidence']]
                        st.download_button(
                            f"⬇️ Download Selected ({len(selected_ids)} CSV)",
                            data=export_selected_df.to_csv(index=False).encode("utf-8"),
                            file_name="selected_heart_ai_records.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                if st.session_state.pop("export_records_hint", False):
                    st.info("Export is ready below.")

                export_df = history_df[['timestamp', 'patient_name', 'age', 'Sex', 'weight', 'prediction_str', 'Confidence']]
                st.download_button(
                    "Download Displayed Records (CSV)",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="filtered_heart_ai_records.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
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
            st.dataframe(df.head(100), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
                st.subheader("condition Distribution")
                target_counts = df['target'].value_counts().reset_index()
                target_counts.columns = ['Status', 'Count']
                fig_pie = px.pie(target_counts, values='Count', names='Status', hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe)
                fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
                st.subheader("Age vs Max Heart Rate")
                fig_scatter = px.scatter(df, x="age", y="thalach", color="target", color_continuous_scale='RdBu')
                fig_scatter.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter, width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Explorer Error: {e}")
