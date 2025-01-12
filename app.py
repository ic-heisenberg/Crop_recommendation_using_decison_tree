import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('decision_tree_model.pkl')

# Feature names
feature_names = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]

# Streamlit app
st.title("Crop Recommendation System")
st.write("Enter the following values to predict the most suitable crop:")

# User input fields
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", min_value=0.0, step=1.0)
    user_input.append(value)

# Convert user input to a numpy array
user_input = np.array(user_input).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(user_input)
    st.success(f"The recommended crop is: {prediction[0]}")
