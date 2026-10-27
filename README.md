# Streamlit ML Project Demos

This is a separate deployment bundle for two portfolio demos. It does **not** modify either original GitHub project repository.

## Contents

- `laptop_app/app.py` — laptop price regression + price-category prediction app.
- `laptop_app/models/best_model.pkl` — trained Random Forest pipeline from the capstone Part 3/4.
- `laptop_app/data/cleaned_data.csv` — cleaned laptop data used for category choices and Ridge regression training.
- `student_app/app.py` — student outcome prediction app that trains the notebook's Random Forest workflow at startup.
- `student_app/data.csv` — original semicolon-delimited student dataset.
- `requirements.txt` — shared dependencies for both apps.

## Deploy both apps with Streamlit Community Cloud

1. Extract this ZIP file.
2. Create **one new GitHub repository** (for example, `streamlit-ml-project-demos`). Do not overwrite your existing project repositories.
3. Upload the extracted bundle contents so `requirements.txt`, `laptop_app/`, and `student_app/` are at the repository root.
4. Open https://share.streamlit.io/ and connect GitHub.
5. Create the first app using this repository and the main file path `laptop_app/app.py`.
6. Create the second app using the same repository and the main file path `student_app/app.py`.
7. Choose Python 3.11 if the deployment settings offer a Python version selector. The laptop classifier was serialized with scikit-learn 1.6.1, which is pinned in `requirements.txt`.

## Optional Gemini explanations

The laptop predictor works without a Gemini API key. To enable the optional explanation button, open the laptop app's Streamlit Cloud app settings, find **Secrets**, and add:

```toml
GEMINI_API_KEY = "your-key-here"
```

Never commit the real API key to GitHub. The demo uses the Gemini model name `gemini-2.5-flash`; if that model is unavailable to your API account, change the model name in `laptop_app/app.py` to one available to you.

## Notes and limitations

- The laptop app estimates price with Ridge regression trained on the included historical dataset and classifies the price as above or at/below that dataset's median using the saved Random Forest pipeline. It is not a live market price service.
- The student app uses the original numeric-coded dataset fields and removes first- and second-semester performance features, matching the feature-selection approach in the notebook. It retrains a Random Forest at startup and reports held-out evaluation metrics.
- The student model is an educational demonstration, not a validated system for real student-risk decisions.
