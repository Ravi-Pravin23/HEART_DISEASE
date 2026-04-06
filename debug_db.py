import sqlite3
import pandas as pd

conn = sqlite3.connect('data/patients.db')
df = pd.read_sql_query("SELECT * FROM patients LIMIT 5", conn)
print("Schema Info:")
c = conn.cursor()
c.execute("PRAGMA table_info(patients)")
for row in c.fetchall():
    print(row)
print("\nFirst 5 rows:")
print(df)
conn.close()
