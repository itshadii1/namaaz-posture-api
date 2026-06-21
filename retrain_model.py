"""
Retrains body_language.pkl from detection.csv using the current sklearn version.
Run once after cloning: python retrain_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV_PATH = Path(__file__).parent.parent / (
    "Namaz-Posture-Identification/AI project/AI Prayer Posture Identification/detection.csv"
)
MODEL_OUT = Path(__file__).parent / "ml" / "body_language.pkl"


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df.columns[~df.isnull().all()]]

    # Normalise class labels to lowercase
    df["class"] = df["class"].str.lower()

    X = df.drop("class", axis=1)
    y = df["class"]

    print(f"Dataset: {len(df)} samples | {X.shape[1]} features")
    print("Class distribution:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1234
    )

    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
