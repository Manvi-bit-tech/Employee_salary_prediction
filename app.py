import streamlit as st
import pandas as pd
import joblib
import pickle

# --- Load All Artifacts ---
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open('label_encoders.pkl', 'rb') as f:
        le_dict = pickle.load(f)
except FileNotFoundError:
    st.error("Error: One or more required files (.pkl) not found. Please run the final notebook.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict an employee's income class based on their profile.")

# --- Sidebar Inputs ---
st.sidebar.header("Enter Employee Details")
# Input fields wahi rahenge...
age = st.sidebar.slider("Age", 17, 90, 35)
workclass = st.sidebar.selectbox("Work Class", le_dict['workclass'].classes_)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=1, value=77516)
educational_num = st.sidebar.slider("Years of Education", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", le_dict['marital-status'].classes_)
occupation = st.sidebar.selectbox("Occupation", le_dict['occupation'].classes_)
race = st.sidebar.selectbox("Race", le_dict['race'].classes_)
gender = st.sidebar.selectbox("Gender", le_dict['gender'].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", le_dict['native-country'].classes_)

# --- Prepare Data for Prediction ---
input_data = {
    'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'educational-num': educational_num,
    'marital-status': marital_status, 'occupation': occupation, 'race': race,
    'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss,
    'hours-per-week': hours_per_week, 'native-country': native_country,
}
input_df = pd.DataFrame([input_data])

# --- CRITICAL STEP: Apply saved Label Encoders ---
for col, le in le_dict.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# --- CRITICAL STEP: Scale the numerical features ---
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Enforce column order to match model's training
expected_features = list(le_dict.keys()) + numerical_cols
# The order must be exactly as it was during training
expected_features = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
input_df = input_df[expected_features]


# --- Prediction ---
if st.button("Predict Salary Class", type="primary"):
    prediction = model.predict(input_df)
    st.subheader("Prediction Result")
    # Assuming 1 means '>50K' and 0 means '<=50K'
    if prediction[0] == 1:
        st.success("This individual is predicted to earn **more than $50K** per year. 🎉")
    else:
        st.info("This individual is predicted to earn **$50K or less** per year.")

