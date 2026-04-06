import sqlite3
import pandas as pd

PATIENTS_DB = 'data/patients.db'

def debug_database():
    print(f"--- Debugging {PATIENTS_DB} ---")
    conn = sqlite3.connect(PATIENTS_DB)
    
    # 1. Check Schema
    print("\n[1] Table Structure (Schema):")
    c = conn.cursor()
    c.execute("PRAGMA table_info(patients)")
    columns = c.fetchall()
    for col in columns:
        print(f"Col {col[0]}: {col[1]} ({col[2]})")
    
    # 2. Check Data
    print("\n[2] Last 5 Patient Records (Full Columns):")
    # Fetch all columns to see where patient_name is
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC LIMIT 5", conn)
    
    if df.empty:
        print("No records found in database.")
    else:
        # Reorder df to show important columns first for readability in console
        all_cols = df.columns.tolist()
        important_cols = ['id', 'timestamp', 'doctor_name', 'patient_name', 'prediction_str']
        other_cols = [c for c in all_cols if c not in important_cols]
        print(df[important_cols + other_cols])
        
    conn.close()

if __name__ == "__main__":
    debug_database()
