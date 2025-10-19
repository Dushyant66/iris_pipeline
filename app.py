import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_PATH = "data/iris.csv"
MODEL_PATH = "models/model.pkl"

def load_data(csv_path=DATA_PATH):
    return pd.read_csv(csv_path)

def validate_data(df):
    if df.isnull().sum().sum() > 0:
        raise ValueError("Data contains missing values")
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"Unexpected columns found. Got {set(df.columns)}")
    return True

def train_model(df):
    X = df.drop("species", axis=1)
    y = df["species"]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)

def predict(model, df):
    X = df.drop("species", axis=1)
    return model.predict(X)

if __name__ == "__main__":
    df = load_data()
    validate_data(df)
    model = train_model(df)
    preds = predict(model, df)
    print(" Model trained successfully!")
    print("Sample Predictions:", preds[:5])
