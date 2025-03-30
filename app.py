import streamlit as st
import joblib
import numpy as np

#Loading the models and scaler
logistic_regression = joblib.load('logistic_regression_model.joblib')
knn = joblib.load('knn_model.joblib')
random_forest = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler_model.joblib')


#Streamlite code
st.title("ML Model Prediction App")
st.sidebar.header("Input Features")


model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    options=["Logistic Regression", "KNN", "Random Forest"]
)
if model_choice == "Logistic Regression":
    model = logistic_regression
elif model_choice == "KNN":
    model = knn
elif model_choice == "Random Forest":
    model = random_forest

feature1 = st.sidebar.text_input("Month", "0.0")
feature2 = st.sidebar.text_input("Age", "0.0")
feature3 = st.sidebar.text_input("Annual Income", "0.0")
feature4 = st.sidebar.text_input("Number of Bank Accounts", "0.0")
feature5 = st.sidebar.text_input("Number of credit cards", "0.0")
feature6 = st.sidebar.text_input("Interest Rate", "0.0")
feature7 = st.sidebar.text_input("Number of Loans", "0.0")
feature8 = st.sidebar.text_input("Delays from due date", "0.0")
feature9 = st.sidebar.text_input("Number of Delayed Payments", "0.0")
feature10 = st.sidebar.text_input("Credit Mix", "0.0")
feature11 = st.sidebar.text_input("Credit History Age", "0.0")
feature12 = st.sidebar.text_input("Monthly Balance", "0.0")

user_inputs = [
    feature1, feature2, feature3, feature4, feature5, feature6,
    feature7, feature8, feature9, feature10, feature11, feature12
]

if st.button("Predict"):
    input_array = np.array(user_inputs).reshape(1,-1)
    scaled_inputs = scaler.transform(input_array)
    prediction = model.predict(scaled_inputs)[0]
    st.write(f"Predicted Result:   {prediction}")