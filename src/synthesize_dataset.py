import pandas as pd
import numpy as np

np.random.seed(42)
SAMPLES_PER_CLASS = 1200  # 1200 x 13 = 15,600 total

def gen(n, age, sex_p, cp, bp, chol, fbs_p, ecg, hr, exang_p, op, slope, ca, thal, weight, label):
    rows = []
    for _ in range(n):
        rows.append({
            'age':      int(np.clip(np.random.normal(age[0], age[1]), 18, 90)),
            'sex':      int(np.random.binomial(1, sex_p)),
            'cp':       int(np.random.choice(cp[0], p=cp[1])),
            'trestbps': int(np.clip(np.random.normal(bp[0], bp[1]), 80, 250)),
            'chol':     int(np.clip(np.random.normal(chol[0], chol[1]), 100, 600)),
            'fbs':      int(np.random.binomial(1, fbs_p)),
            'restecg':  int(np.random.choice(ecg[0], p=ecg[1])),
            'thalach':  int(np.clip(np.random.normal(hr[0], hr[1]), 50, 250)),
            'exang':    int(np.random.binomial(1, exang_p)),
            'oldpeak':  round(np.clip(np.random.normal(op[0], op[1]), 0.0, 10.0), 1),
            'slope':    int(np.random.choice(slope[0], p=slope[1])),
            'ca':       int(np.random.choice(ca[0], p=ca[1])),
            'thal':     int(np.random.choice(thal[0], p=thal[1])),
            'weight':   int(np.clip(np.random.normal(weight[0], weight[1]), 40, 200)),
            'target':   label
        })
    return rows

all_data = []

# ── Class 0: Healthy ─────────────────────────────────────────────────────────
# Distinctive: young-ish, normal BP/chol, no exang, low oldpeak, 0 vessels, thal=1
all_data += gen(SAMPLES_PER_CLASS,
    age=(44, 8),  sex_p=0.50, cp=([0,1,2,3],[0.50,0.25,0.15,0.10]),
    bp=(112, 8),  chol=(185, 22), fbs_p=0.05,
    ecg=([0,1,2],[0.85,0.12,0.03]), hr=(170, 15), exang_p=0.03,
    op=(0.2, 0.2), slope=([0,1,2],[0.65,0.28,0.07]),
    ca=([0,1,2,3],[0.92,0.05,0.02,0.01]), thal=([1,2,3],[0.80,0.15,0.05]),
    weight=(72, 10), label=0)

# ── Class 1: Coronary Artery Disease (CAD) ───────────────────────────────────
# Distinctive: mid-age male, high BP, high chol, exang, moderate oldpeak, reversible defect
all_data += gen(SAMPLES_PER_CLASS,
    age=(58, 7),  sex_p=0.80, cp=([0,1,2,3],[0.08,0.15,0.32,0.45]),
    bp=(145, 14), chol=(265, 35), fbs_p=0.30,
    ecg=([0,1,2],[0.20,0.55,0.25]), hr=(135, 18), exang_p=0.65,
    op=(2.2, 0.9), slope=([0,1,2],[0.08,0.27,0.65]),
    ca=([0,1,2,3],[0.12,0.28,0.38,0.22]), thal=([1,2,3],[0.05,0.20,0.75]),
    weight=(88, 14), label=1)

# ── Class 2: Heart Attack (Myocardial Infarction) ────────────────────────────
# Distinctive: older, very high BP, very high chol, asymptomatic cp, very high oldpeak, 3+ vessels
all_data += gen(SAMPLES_PER_CLASS,
    age=(62, 7),  sex_p=0.82, cp=([0,1,2,3],[0.03,0.07,0.10,0.80]),
    bp=(162, 16), chol=(290, 40), fbs_p=0.38,
    ecg=([0,1,2],[0.05,0.20,0.75]), hr=(112, 18), exang_p=0.88,
    op=(4.5, 1.2), slope=([0,1,2],[0.02,0.13,0.85]),
    ca=([0,1,2,3],[0.05,0.12,0.28,0.55]), thal=([1,2,3],[0.02,0.12,0.86]),
    weight=(94, 15), label=2)

# ── Class 3: Arrhythmia ──────────────────────────────────────────────────────
# Distinctive: moderate age, ECG abnormality dominant, high HR variation, low exang
all_data += gen(SAMPLES_PER_CLASS,
    age=(50, 10), sex_p=0.56, cp=([0,1,2,3],[0.35,0.32,0.20,0.13]),
    bp=(124, 12), chol=(215, 32), fbs_p=0.12,
    ecg=([0,1,2],[0.05,0.75,0.20]), hr=(158, 28), exang_p=0.12,
    op=(0.8, 0.6), slope=([0,1,2],[0.42,0.42,0.16]),
    ca=([0,1,2,3],[0.68,0.20,0.09,0.03]), thal=([1,2,3],[0.55,0.32,0.13]),
    weight=(78, 12), label=3)

# ── Class 4: Heart Failure (Congestive) ──────────────────────────────────────
# Distinctive: elderly, high BP, moderate chol, high oldpeak, multiple vessels, LV hypertrophy ECG
all_data += gen(SAMPLES_PER_CLASS,
    age=(66, 7),  sex_p=0.68, cp=([0,1,2,3],[0.15,0.18,0.32,0.35]),
    bp=(152, 16), chol=(242, 38), fbs_p=0.32,
    ecg=([0,1,2],[0.08,0.30,0.62]), hr=(110, 18), exang_p=0.72,
    op=(3.2, 1.1), slope=([0,1,2],[0.05,0.22,0.73]),
    ca=([0,1,2,3],[0.10,0.22,0.38,0.30]), thal=([1,2,3],[0.05,0.28,0.67]),
    weight=(82, 14), label=4)

# ── Class 5: Heart Valve Disease ─────────────────────────────────────────────
# Distinctive: mid-late age, atypical angina cp, moderate ECG, moderate HR drop, fixed defect
all_data += gen(SAMPLES_PER_CLASS,
    age=(55, 9),  sex_p=0.54, cp=([0,1,2,3],[0.15,0.55,0.20,0.10]),
    bp=(130, 13), chol=(232, 35), fbs_p=0.15,
    ecg=([0,1,2],[0.22,0.60,0.18]), hr=(138, 22), exang_p=0.38,
    op=(1.5, 0.8), slope=([0,1,2],[0.22,0.52,0.26]),
    ca=([0,1,2,3],[0.35,0.42,0.16,0.07]), thal=([1,2,3],[0.15,0.72,0.13]),
    weight=(80, 12), label=5)

# ── Class 6: Cardiomyopathy ──────────────────────────────────────────────────
# Distinctive: younger, non-anginal cp, LV hypertrophy ECG, significant oldpeak, low CA
all_data += gen(SAMPLES_PER_CLASS,
    age=(48, 10), sex_p=0.62, cp=([0,1,2,3],[0.22,0.18,0.48,0.12]),
    bp=(132, 14), chol=(222, 33), fbs_p=0.16,
    ecg=([0,1,2],[0.10,0.28,0.62]), hr=(122, 20), exang_p=0.48,
    op=(2.5, 1.0), slope=([0,1,2],[0.10,0.38,0.52]),
    ca=([0,1,2,3],[0.48,0.32,0.14,0.06]), thal=([1,2,3],[0.12,0.48,0.40]),
    weight=(84, 15), label=6)

# ── Class 7: Congenital Heart Defects ────────────────────────────────────────
# Distinctive: YOUNG age, normal BP/chol, no FBS, normal/mild ECG, high HR
all_data += gen(SAMPLES_PER_CLASS,
    age=(32, 9),  sex_p=0.52, cp=([0,1,2,3],[0.35,0.32,0.22,0.11]),
    bp=(115, 9),  chol=(188, 25), fbs_p=0.06,
    ecg=([0,1,2],[0.42,0.48,0.10]), hr=(172, 22), exang_p=0.18,
    op=(0.7, 0.5), slope=([0,1,2],[0.55,0.35,0.10]),
    ca=([0,1,2,3],[0.72,0.20,0.06,0.02]), thal=([1,2,3],[0.42,0.40,0.18]),
    weight=(68, 10), label=7)

# ── Class 8: Pericarditis ────────────────────────────────────────────────────
# Distinctive: younger, sharp cp (typical angina 0), ST-T ECG changes, normal-low oldpeak
all_data += gen(SAMPLES_PER_CLASS,
    age=(40, 11), sex_p=0.60, cp=([0,1,2,3],[0.68,0.18,0.10,0.04]),
    bp=(118, 10), chol=(198, 28), fbs_p=0.08,
    ecg=([0,1,2],[0.08,0.80,0.12]), hr=(155, 22), exang_p=0.10,
    op=(0.6, 0.5), slope=([0,1,2],[0.55,0.38,0.07]),
    ca=([0,1,2,3],[0.75,0.18,0.05,0.02]), thal=([1,2,3],[0.60,0.30,0.10]),
    weight=(74, 10), label=8)

# ── Class 9: Myocarditis ─────────────────────────────────────────────────────
# Distinctive: youngest adult, same sharp cp, LV hypertrophy ECG, very high HR
all_data += gen(SAMPLES_PER_CLASS,
    age=(34, 9),  sex_p=0.58, cp=([0,1,2,3],[0.60,0.22,0.13,0.05]),
    bp=(114, 9),  chol=(192, 26), fbs_p=0.07,
    ecg=([0,1,2],[0.05,0.35,0.60]), hr=(172, 25), exang_p=0.08,
    op=(0.5, 0.4), slope=([0,1,2],[0.58,0.35,0.07]),
    ca=([0,1,2,3],[0.80,0.14,0.04,0.02]), thal=([1,2,3],[0.65,0.26,0.09]),
    weight=(70, 10), label=9)

# ── Class 10: Endocarditis ───────────────────────────────────────────────────
# Distinctive: moderate age, fever→high FBS, ST-T changes, moderate HR, fixed defect
all_data += gen(SAMPLES_PER_CLASS,
    age=(46, 11), sex_p=0.65, cp=([0,1,2,3],[0.30,0.32,0.25,0.13]),
    bp=(122, 12), chol=(208, 30), fbs_p=0.42,
    ecg=([0,1,2],[0.12,0.68,0.20]), hr=(145, 24), exang_p=0.20,
    op=(1.0, 0.7), slope=([0,1,2],[0.42,0.42,0.16]),
    ca=([0,1,2,3],[0.58,0.28,0.10,0.04]), thal=([1,2,3],[0.28,0.58,0.14]),
    weight=(76, 12), label=10)

# ── Class 11: Aortic Aneurysm ────────────────────────────────────────────────
# Distinctive: elderly male, VERY high BP, high chol, high FBS, non-anginal cp
all_data += gen(SAMPLES_PER_CLASS,
    age=(67, 7),  sex_p=0.78, cp=([0,1,2,3],[0.12,0.18,0.52,0.18]),
    bp=(168, 14), chol=(252, 38), fbs_p=0.35,
    ecg=([0,1,2],[0.30,0.45,0.25]), hr=(128, 18), exang_p=0.42,
    op=(1.8, 0.9), slope=([0,1,2],[0.22,0.42,0.36]),
    ca=([0,1,2,3],[0.25,0.35,0.28,0.12]), thal=([1,2,3],[0.18,0.42,0.40]),
    weight=(92, 15), label=11)

# ── Class 12: Peripheral Artery Disease (PAD) ────────────────────────────────
# Distinctive: elderly, very high chol, diabetic FBS, exercise angina, asymptomatic cp
all_data += gen(SAMPLES_PER_CLASS,
    age=(63, 8),  sex_p=0.68, cp=([0,1,2,3],[0.10,0.15,0.28,0.47]),
    bp=(148, 14), chol=(278, 38), fbs_p=0.45,
    ecg=([0,1,2],[0.35,0.42,0.23]), hr=(128, 18), exang_p=0.62,
    op=(2.2, 1.0), slope=([0,1,2],[0.15,0.38,0.47]),
    ca=([0,1,2,3],[0.18,0.30,0.35,0.17]), thal=([1,2,3],[0.10,0.32,0.58]),
    weight=(88, 14), label=12)

# ── Build final dataset ────────────────────────────────────────────────────────
df = pd.DataFrame(all_data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/heart.csv', index=False)

print(f"✅ Synthetic 13-class dataset created: {len(df)} total samples")
print("\nClass Distribution:")
names = {
    0:'Healthy', 1:'CAD', 2:'Heart Attack', 3:'Arrhythmia',
    4:'Heart Failure', 5:'Valve Disease', 6:'Cardiomyopathy',
    7:'Congenital Defects', 8:'Pericarditis', 9:'Myocarditis',
    10:'Endocarditis', 11:'Aortic Aneurysm', 12:'PAD'
}
for cls, count in df['target'].value_counts().sort_index().items():
    print(f"  Class {cls} ({names[cls]}): {count} samples")
