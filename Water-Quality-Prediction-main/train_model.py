# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import os

# 🔍 Step 1: Check if dataset exists
dataset_name = "water_potability.csv"  # <-- change this if your file has a different name

if not os.path.exists(dataset_name):
    print(f"❌ Dataset file '{dataset_name}' not found! Put your CSV file in this folder.")
else:
    # ✅ Step 2: Load dataset
    data = pd.read_csv(dataset_name)
    print("✅ Dataset loaded successfully!")

    # ✅ Step 3: Prepare data
    X = data.drop("Potability", axis=1)
    y = data["Potability"]
    X = X.fillna(X.mean())

    # ✅ Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Step 5: Train model
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    print("✅ Model trained successfully!")

    # ✅ Step 6: Save model
    joblib.dump(svm, "svm.pkl")
    print("✅ Model saved as svm.pkl")
