import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Simple cleaning: Remove duplicates
    df = df.drop_duplicates()
    
    # Define features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data (Crucial for high accuracy)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, X.columns

print("Data Engine script ready!")