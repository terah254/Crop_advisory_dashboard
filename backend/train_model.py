import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_save():
    print("Training model...")
    np.random.seed(42)
    N = 2000

    temp_max      = np.random.uniform(15, 40, N)
    temp_min      = np.random.uniform(5, 25, N)
    precipitation = np.random.exponential(5, N)
    precip_prob   = np.random.uniform(0, 100, N)
    wind_max      = np.random.uniform(0, 60, N)

    risk = np.zeros(N, dtype=int)
    risk[temp_max > 35]      += 1
    risk[temp_max > 38]      += 1
    risk[temp_min < 8]       += 1
    risk[precipitation > 40] += 1
    risk[precipitation > 60] += 1
    risk[precip_prob > 80]   += 1
    risk[wind_max > 45]      += 1
    risk = np.clip(risk, 0, 2)

    X = np.column_stack([temp_max, temp_min, precipitation, precip_prob, wind_max])
    y = risk

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/risk_model.pkl")
    print("Model saved to models/risk_model.pkl")

if __name__ == "__main__":
    train_and_save()