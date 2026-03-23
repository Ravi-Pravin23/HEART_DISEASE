import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from src.data_engine import prepare_data

NUM_CLASSES = 13

# 1. Load & Prepare Data
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'heart.csv')
X_train, X_test, y_train, y_test, scaler, columns = prepare_data(data_path)

# Stack all data for CV
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

# 2. Train XGBoost with Hyperparameter Tuning (GridSearchCV)
base_model = XGBClassifier(
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1                  # Use all CPU cores
)

# Define the hyperparameter grid to search over
param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6]
}

print("🚀 Running GridSearchCV for AI Tuning (testing multiple configurations)...")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"\n🏆 Best Hyperparameters Found: {grid_search.best_params_}")
model = grid_search.best_estimator_

# 3. Evaluate on test set
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"\n🔥 Test Accuracy: {acc * 100:.2f}%")

disease_names = [
    'Healthy', 'CAD', 'Heart Attack', 'Arrhythmia', 'Heart Failure',
    'Valve Disease', 'Cardiomyopathy', 'Congenital Defects',
    'Pericarditis', 'Myocarditis', 'Endocarditis', 'Aortic Aneurysm', 'PAD'
]
print("\n📊 Per-Class Report:")
print(classification_report(y_test, predictions, target_names=disease_names))

# 4. Cross-Validation
print("⚡ Running 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='accuracy', n_jobs=-1)
print(f"✅ CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# 5. Save Model
model_path = os.path.join(BASE_DIR, 'models', 'heart_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
features_path = os.path.join(BASE_DIR, 'models', 'features.pkl')
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(list(columns), features_path)
print("\n✅ Model saved: models/heart_model.pkl, models/scaler.pkl, models/features.pkl")
