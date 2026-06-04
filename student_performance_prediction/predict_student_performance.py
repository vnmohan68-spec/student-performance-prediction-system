import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "student_performance.csv"
MODEL_PATH = BASE_DIR / "student_performance_model.joblib"
SCALER_PATH = BASE_DIR / "student_performance_scaler.joblib"


def generate_synthetic_dataset(path: Path, n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    study_hours = np.round(np.random.uniform(0.0, 10.0, size=n_samples), 1)
    attendance_rate = np.round(np.random.uniform(40.0, 100.0, size=n_samples), 1)
    previous_score = np.round(np.random.uniform(30.0, 100.0, size=n_samples), 1)

    logits = 0.35 * study_hours + 0.03 * attendance_rate + 0.04 * previous_score - 5.5
    pass_probability = 1 / (1 + np.exp(-logits))
    pass_fail = np.where(pass_probability > np.random.rand(n_samples), "pass", "fail")

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance_rate": attendance_rate,
        "previous_score": previous_score,
        "pass_fail": pass_fail,
    })
    df.to_csv(path, index=False)
    return df


def load_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded dataset from {path}")
    else:
        print(f"Dataset not found at {path}. Generating synthetic dataset...")
        df = generate_synthetic_dataset(path)
        print(f"Synthetic dataset saved to {path}")
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    print("\n=== Dataset Overview ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nHead of dataset:")
    print(df.head(10).to_string(index=False))
    print("\nData types:")
    print(df.dtypes)
    print("\nTarget distribution:")
    print(df["pass_fail"].value_counts())
    print("\nSummary statistics:")
    print(df.describe().round(2).T)


def preprocess_dataset(df: pd.DataFrame):
    X = df[["study_hours", "attendance_rate", "previous_score"]].copy()
    y = df["pass_fail"].map({"fail": 0, "pass": 1}).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["fail", "pass"]))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_artifacts(model: RandomForestClassifier, scaler: StandardScaler) -> None:
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


def predict_student(model: RandomForestClassifier, scaler: StandardScaler, study_hours: float, attendance_rate: float, previous_score: float) -> str:
    features = np.array([[study_hours, attendance_rate, previous_score]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return "pass" if prediction == 1 else "fail"


def main() -> None:
    df = load_dataset(DATA_PATH)
    explore_dataset(df)

    X_train, X_test, y_train, y_test, scaler = preprocess_dataset(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, scaler)

    sample = {
        "study_hours": 5.5,
        "attendance_rate": 82.0,
        "previous_score": 78.0,
    }
    outcome = predict_student(
        model,
        scaler,
        sample["study_hours"],
        sample["attendance_rate"],
        sample["previous_score"],
    )
    print("\n=== Sample Prediction ===")
    print(f"Input: {sample}")
    print(f"Predicted outcome: {outcome}")


if __name__ == "__main__":
    main()
