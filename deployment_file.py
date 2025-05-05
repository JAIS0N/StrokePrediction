# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from subdirectory.preprocess import preprocess_input
from model_router import load_model

st.set_page_config(page_title="Stroke Predictor", layout="centered")
st.title("🧠 Stroke Prediction App")

# -------------------------
# 1. Model Selection
# -------------------------
model_options = [
    "randomforest", "xgboost", "catboost", "decisiontree",
    "balancedrandomforest", "lightgbm", "knn", "svm",
    "naivebayes", "logisticregression", "votingclassifier", "stacking"
]
model_choice = st.selectbox("Choose a Model to Use:", model_options)

# -------------------------
# 2. User Input
# -------------------------
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 0, 100)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", 0.0, 300.0)
    bmi = st.number_input("BMI", 10.0, 60.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("Predict")

# -------------------------
# 3. Prediction Logic
# -------------------------
if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    try:
        X_processed = preprocess_input(input_data, model_choice)
        model = load_model(model_choice)
        prediction = model.predict(X_processed)[0]

        st.success(f"✅ Prediction: {'Stroke Risk' if prediction == 1 else 'No Stroke Risk'}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
