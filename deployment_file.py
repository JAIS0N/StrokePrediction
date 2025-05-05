import streamlit as st
import pickle
import numpy as np

# Load model
with open("D:/Studies/machine_learning_data_set/StrokePrediction/models/logistic.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Stroke Prediction App")

st.write("Enter patient details below:")

# Numerical inputs
age = st.number_input("Age", min_value=0.0, max_value=120.0)
avg_glucose = st.number_input("Average Glucose Level")
bmi = st.number_input("BMI")

# Binary inputs
hypertension = st.checkbox("Hypertension")
heart_disease = st.checkbox("Heart Disease")

# One-hot encoded categories
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Govt_job", "Never_worked", "Private", "Self-employed", "children"])
residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
smoking_status = st.selectbox("Smoking Status", ["Unknown", "formerly smoked", "never smoked", "smokes"])

# Manual one-hot encoding (matching model input format)
gender_Male = gender == "Male"
gender_Other = gender == "Other"
ever_married_Yes = ever_married == "Yes"

work_type_Never_worked = work_type == "Never_worked"
work_type_Private = work_type == "Private"
work_type_Self_employed = work_type == "Self-employed"
work_type_children = work_type == "children"

Residence_type_Urban = residence_type == "Urban"

smoking_status_formerly = smoking_status == "formerly smoked"
smoking_status_never = smoking_status == "never smoked"
smoking_status_smokes = smoking_status == "smokes"

# Assemble features in the same order as training
features = np.array([[age, hypertension, heart_disease, avg_glucose, bmi,
                      gender_Male, gender_Other, ever_married_Yes,
                      work_type_Never_worked, work_type_Private,
                      work_type_Self_employed, work_type_children,
                      Residence_type_Urban,
                      smoking_status_formerly, smoking_status_never, smoking_status_smokes]])

# Predict
if st.button("Predict Stroke Risk"):
    result = model.predict(features)
    print(result)
    proba = model.predict_proba(features)
    st.success(f"Prediction: {'Stroke Risk' if result[0] == 1 else 'No Stroke Risk'}")

