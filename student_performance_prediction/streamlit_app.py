import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "student_performance.csv"
MODEL_PATH = BASE_DIR / "student_performance_model.joblib"
SCALER_PATH = BASE_DIR / "student_performance_scaler.joblib"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def preprocess_dataset(df: pd.DataFrame):
    X = df[["study_hours", "attendance_rate", "previous_score"]]
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


def load_or_train_model(df: pd.DataFrame):
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        X_train, X_test, y_train, y_test, scaler = preprocess_dataset(df)
        model = train_model(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    return model, scaler


def evaluate_model(model: RandomForestClassifier, scaler: StandardScaler, df: pd.DataFrame):
    X = df[["study_hours", "attendance_rate", "previous_score"]].copy()
    y = df["pass_fail"].map({"fail": 0, "pass": 1}).astype(int)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "classification_report": classification_report(y, y_pred, target_names=["fail", "pass"], output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }


def make_prediction(model: RandomForestClassifier, scaler: StandardScaler, study_hours: float, attendance_rate: float, previous_score: float):
    features = np.array([[study_hours, attendance_rate, previous_score]])
    features_scaled = scaler.transform(features)
    label = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][label]
    return "pass" if label == 1 else "fail", float(probability)


def main():
    st.title("Student Performance Prediction")
    st.write("Predict whether a student will pass or fail based on study hours, attendance, and previous score.")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}. Run the training script first to generate it.")
        st.stop()

    df = load_dataset(DATA_PATH)
    st.sidebar.header("Student Inputs")
    study_hours = st.sidebar.slider("Study hours per day", 0.0, 10.0, 5.0, 0.1)
    attendance_rate = st.sidebar.slider("Attendance rate (%)", 40.0, 100.0, 75.0, 0.1)
    previous_score = st.sidebar.slider("Previous exam score", 30.0, 100.0, 65.0, 0.1)

    model, scaler = load_or_train_model(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    st.write(f"Total records: {len(df)}")
    st.write(df["pass_fail"].value_counts())

    if st.button("Predict"):
        label, prob = make_prediction(model, scaler, study_hours, attendance_rate, previous_score)
        st.success(f"Predicted result: **{label.upper()}**")
        st.write(f"Confidence: {prob:.2%}")

    st.markdown("---")
    st.subheader("Model Evaluation")
    metrics = evaluate_model(model, scaler, df)
    st.write(f"Accuracy: **{metrics['accuracy']:.2f}**")
    st.write("Confusion matrix:")
    st.write(metrics["confusion_matrix"])
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    st.write(report)

    st.markdown("---")
    st.write("This Streamlit app uses the same student performance dataset and a Random Forest classifier.")


if __name__ == "__main__":
    main()
