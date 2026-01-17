import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("notebooks/svm_diabetes_model.pkl", "rb"))
scaler = pickle.load(open("notebooks/scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")

st.title("🩺 Diabetes Prediction")
st.markdown(
    """
    This app predicts whether a person is likely to have diabetes based on medical parameters.
    """
)

with st.form(key="diabetes_form"):
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=6)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=148)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=72)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=35)
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=33.6)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.627)
    age = st.number_input("Age", min_value=1, max_value=120, value=50)

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    prediction_proba = model.decision_function(input_scaled)


    if prediction[0] == 1:
        st.error(f"The person is likely diabetic. (Score: {prediction_proba[0]:.2f})")
    else:
        st.success(f"The person is not diabetic. (Score: {prediction_proba[0]:.2f})")