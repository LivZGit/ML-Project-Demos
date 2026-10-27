from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "cleaned_data.csv"
MODEL_PATH = APP_DIR / "models" / "best_model.pkl"

st.set_page_config(page_title="Laptop Price Intelligence", page_icon="💻", layout="wide")
st.title("💻 Laptop Price Intelligence")
st.caption("Laptop price estimation and price-category classification using your capstone dataset.")
st.info("This is an academic demonstration trained on historical laptop listings. Predictions are estimates, not live market quotes.")

@st.cache_resource
def load_assets():
    df = pd.read_csv(DATA_PATH)
    classifier = joblib.load(MODEL_PATH)
    raw_x = df.drop(columns=["Price"]).copy()
    y_price = df["Price"].astype(float)
    cat_cols = raw_x.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded = pd.get_dummies(raw_x, columns=cat_cols, drop_first=True, dtype=int)
    feature_cols = list(getattr(classifier, "feature_names_in_", encoded.columns))
    encoded = encoded.reindex(columns=feature_cols, fill_value=0)
    X_train, X_test, y_train, y_test = train_test_split(
        encoded, y_price, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train_scaled, y_train)
    r2 = r2_score(y_test, regressor.predict(X_test_scaled))
    return df, raw_x, cat_cols, feature_cols, classifier, scaler, regressor, r2

try:
    df, raw_x, cat_cols, feature_cols, classifier, scaler, regressor, r2 = load_assets()
except Exception as exc:
    st.error("The app could not load its model or data. Check that the bundled model and dataset files are present and compatible with the dependencies.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("About this demo")
    st.write("**Dataset:** 1,274 cleaned laptop records")
    st.write("**Regression:** Ridge Regression")
    st.write("**Classification:** tuned Random Forest pipeline")
    st.write(f"**Regression test R²:** {r2:.4f}")
    st.caption("The classifier predicts whether a laptop's price is above the dataset median; it does not classify current market prices.")

st.subheader("Enter laptop specifications")
options = {}
for col in cat_cols:
    vals = [v for v in df[col].dropna().unique().tolist()]
    vals = sorted(vals, key=lambda x: str(x))
    options[col] = vals

with st.form("laptop_form"):
    left, right = st.columns(2)
    record = {}
    with left:
        company_opts = options.get("Company", [])
        record["Company"] = st.selectbox("Brand", company_opts, index=company_opts.index("Dell") if "Dell" in company_opts else 0)
        type_opts = options.get("TypeName", [])
        record["TypeName"] = st.selectbox("Laptop type", type_opts, index=type_opts.index("Notebook") if "Notebook" in type_opts else 0)
        record["Inches"] = st.number_input("Screen size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
        record["ScreenResolution"] = st.selectbox("Screen resolution", options.get("ScreenResolution", []))
        record["Cpu"] = st.selectbox("CPU", options.get("Cpu", []))
    with right:
        record["Ram"] = st.selectbox("RAM (GB)", sorted([int(v) for v in df["Ram"].dropna().unique()]), index=(sorted([int(v) for v in df["Ram"].dropna().unique()]).index(8) if 8 in df["Ram"].values else 0))
        record["Memory"] = st.selectbox("Storage", options.get("Memory", []))
        record["Gpu"] = st.selectbox("GPU", options.get("Gpu", []))
        record["OpSys"] = st.selectbox("Operating system", options.get("OpSys", []))
        record["Weight"] = st.number_input("Weight (kg)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    submitted = st.form_submit_button("Predict laptop price", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([record], columns=raw_x.columns)
    # Build dummy columns together with the training data so category encoding
    # follows the same reference categories used in the notebook.
    combined = pd.concat([raw_x, input_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True, dtype=int)
    row_encoded = combined_encoded.iloc[[-1]].reindex(columns=feature_cols, fill_value=0)

    try:
        estimated_price = float(regressor.predict(scaler.transform(row_encoded))[0])
        class_pred = int(classifier.predict(row_encoded)[0])
        probabilities = classifier.predict_proba(row_encoded)[0]
        class_labels = {0: "At or below dataset median", 1: "Above dataset median"}
        label = class_labels.get(class_pred, str(class_pred))
        confidence = float(probabilities[class_pred])

        st.divider()
        st.subheader("Prediction results")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Estimated price", f"₹{estimated_price:,.0f}")
            st.caption("Ridge regression estimate; based on the historical dataset.")
        with c2:
            st.metric("Predicted price category", label)
            st.caption(f"Model probability for this category: {confidence:.1%}")
        st.write("**Category probabilities**")
        st.bar_chart(pd.DataFrame({"Probability": probabilities}, index=["At/below median", "Above median"]))

        st.subheader("AI-generated explanation (optional)")
        st.caption("This feature requires a Gemini API key configured in Streamlit Cloud secrets. The prediction works without it.")
        if st.button("Generate Gemini explanation"):
            api_key = None
            try:
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.warning("No Gemini API key is configured. Add GEMINI_API_KEY in the app's Streamlit secrets to enable explanations.")
            else:
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    prompt = {
                        "laptop_specifications": record,
                        "estimated_price_in_dataset_currency": round(estimated_price, 2),
                        "predicted_price_category": label,
                        "category_probability": round(confidence, 4),
                        "instruction": "Explain this machine-learning output in concise, cautious language. Do not claim these are live market prices or imply causal feature importance. Return JSON with keys: summary, caveats, next_step."
                    }
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=json.dumps(prompt),
                        config={"response_mime_type": "application/json", "temperature": 0.2},
                    )
                    parsed = json.loads(response.text)
                    st.write(parsed.get("summary", ""))
                    if parsed.get("caveats"):
                        st.caption("Caveats: " + str(parsed["caveats"]))
                    if parsed.get("next_step"):
                        st.write("**Suggested next step:**", parsed["next_step"])
                except Exception as exc:
                    st.error("Gemini explanation could not be generated. The ML prediction above is still available.")
                    st.caption(str(exc))
    except Exception as exc:
        st.error("Prediction failed for this input. Please try another combination.")
        st.exception(exc)

st.divider()
st.caption("Built as a portfolio demo. Model predictions are not financial advice or guaranteed market prices.")
