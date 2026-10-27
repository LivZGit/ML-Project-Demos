from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "student_data" / "data.csv"
CONTINUOUS_FEATURES = {
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Dropout & Academic Success Predictor")
st.caption("A Random Forest demo based on the academic-success dataset and the feature-selection approach in the project notebook.")
st.warning("This is an educational model, not a validated student-risk assessment tool. Do not use predictions alone to make decisions about students.")

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [str(c).replace("\t", " ").strip() for c in df.columns]
    df["Target"] = df["Target"].astype(str).str.strip()
    X = df.drop(columns=["Target"]).copy()
    y = df["Target"].copy()
    # Match the notebook's feature selection: remove first- and second-semester
    # performance columns, leaving the 24 pre-/non-semester features.
    semester_features = [c for c in X.columns if "Curricular units 1st sem" in c or "Curricular units 2nd sem" in c]
    X = X.drop(columns=semester_features)
    X = X.apply(pd.to_numeric, errors="coerce")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, pred)
    report = classification_report(
        y_test, pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, pred)
    return df, X, y, label_encoder, model, accuracy, report, cm, X_test, y_test, pred, probs

try:
    df, X, y, label_encoder, model, accuracy, report, cm, X_test, y_test, test_pred, test_probs = train_model()
except Exception as exc:
    st.error("The model could not be trained from the included dataset. Check the data file and dependencies.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Model overview")
    st.write(f"**Dataset records:** {len(df):,}")
    st.write(f"**Input features:** {X.shape[1]}")
    st.write("**Algorithm:** Random Forest (100 trees)")
    st.metric("Held-out test accuracy", f"{accuracy:.2%}")
    st.caption("Test split: 20%, stratified, random_state=42.")

st.subheader("Enter student information")
st.caption("The source dataset represents categorical attributes as numeric codes. The dropdowns below retain those original codes because the repository does not include a complete human-readable codebook.")

# Group fields for a less overwhelming form.
columns = list(X.columns)
groups = {
    "Application and background": [c for c in columns if c in ["Marital status", "Application mode", "Application order", "Course", "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)", "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]],
    "Personal and financial factors": [c for c in columns if c in ["Admission grade", "Displaced", "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International"]],
    "Economic indicators": [c for c in columns if c in ["Unemployment rate", "Inflation rate", "GDP"]],
}
used = {c for group in groups.values() for c in group}
remaining = [c for c in columns if c not in used]
if remaining:
    groups["Other input fields"] = remaining

with st.form("student_form"):
    record = {}
    for group_name, group_cols in groups.items():
        with st.expander(group_name, expanded=(group_name == "Application and background")):
            controls = st.columns(2)
            for i, col in enumerate(group_cols):
                series = X[col].dropna()
                with controls[i % 2]:
                    if col in CONTINUOUS_FEATURES:
                        median = float(series.median())
                        minimum = float(series.min())
                        maximum = float(series.max())
                        step = 0.1 if col in {"Previous qualification (grade)", "Admission grade", "Unemployment rate", "Inflation rate", "GDP"} else 1.0
                        record[col] = st.number_input(
                            col, min_value=minimum, max_value=maximum, value=min(max(median, minimum), maximum), step=step, format="%.2f"
                        )
                    else:
                        values = sorted(series.unique().tolist())
                        try:
                            default_value = series.mode().iloc[0]
                            default_index = values.index(default_value)
                        except Exception:
                            default_index = 0
                        record[col] = st.selectbox(col, values, index=default_index, format_func=lambda v: str(int(v)) if float(v).is_integer() else str(v))
    submitted = st.form_submit_button("Predict student outcome", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{col: record[col] for col in X.columns}], columns=X.columns)
    try:
        pred_id = int(model.predict(row)[0])
        pred_label = str(label_encoder.inverse_transform([pred_id])[0])
        proba = model.predict_proba(row)[0]
        st.divider()
        st.subheader("Prediction result")
        st.metric("Predicted outcome", pred_label)
        st.write("**Model probability by outcome**")
        prob_df = pd.DataFrame({"Probability": proba}, index=label_encoder.classes_).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df)
        st.caption("Probabilities reflect this model's output and should not be interpreted as calibrated real-world student risk.")
    except Exception as exc:
        st.error("Prediction failed for this input.")
        st.exception(exc)

st.divider()
st.subheader("Held-out evaluation")
metric_cols = st.columns(3)
for col, label in zip(metric_cols, label_encoder.classes_):
    col.metric(f"{label} recall", f"{report[label]['recall']:.1%}")
with st.expander("View classification report"):
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(3), use_container_width=True)
with st.expander("View confusion matrix"):
    st.dataframe(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).rename_axis("Actual").rename_axis("Predicted", axis="columns"), use_container_width=True)

st.caption("Portfolio demonstration only. Predictions should not replace human judgment, student support, or institutional review.")
