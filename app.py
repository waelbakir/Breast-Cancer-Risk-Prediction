import streamlit as st
import pandas as pd
import joblib

# Load the trained model and model columns
model = joblib.load("cancer_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Function to preprocess user input
def preprocess_input(user_data):
    data = pd.DataFrame([user_data])
    data = pd.get_dummies(data, drop_first=True)

    # Ensure all columns from training are present
    for col in model_columns:
        if col not in data.columns:
            data[col] = 0  # Add missing columns with 0 values

    # Ensure same column order as in the training data
    data = data[model_columns]
    return data

# Streamlit UI
st.title("Cancer Risk Prediction")

# Collect user input
user_data = {
    "menopaus": st.selectbox("Menopausal Status", [0, 1], format_func=lambda x: "pre" if x == 0 else "post"),
    "agegrp": st.selectbox("Age Group", list(range(1, 11)), format_func=lambda x: f"{30 + (x - 1) * 5}-{34 + (x - 1) * 5}"),
    "density": st.selectbox("Breast Density", [1, 2, 3, 4]),
    "race": st.selectbox("Race", [1, 2, 3], format_func=lambda x: ["White", "Asian", "Black"][x - 1]),
    "Hispanic": st.selectbox("Are you Hispanic?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "bmi": st.number_input("BMI", min_value=0.0, format="%.2f"),
    "agefirst": st.selectbox("Age at First Birth", [0, 1, 2], format_func=lambda x: ["<30", ">=30", "Nulliparous"][x]),
    "nrelbc": st.number_input("Number of Relatives with Breast Cancer", min_value=0, max_value=2, step=1),
    "brstproc": st.selectbox("Breast Procedure?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "lastmamm": st.selectbox("Last Mammogram?", [0, 1], format_func=lambda x: "Negative" if x == 0 else "False Positive"),
    "surgmeno": st.selectbox("Surgical Menopause?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "hrt": st.selectbox("Hormone Replacement Therapy?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "invasive": st.selectbox("History of Invasive Breast Cancer?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "training": st.selectbox("Training Set?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
    "count": st.number_input("Count", min_value=0, step=1),
}

# Button to make prediction
if st.button("Predict"):
    processed_input = preprocess_input(user_data)
    
    # Get raw probability
    prediction_proba = model.predict_proba(processed_input)[0][1]

    # Adjust decision threshold
    threshold = 0.3  # Example: set threshold to 30% for predicting high-risk cancer
    prediction = (prediction_proba > threshold).astype(int)

    # Display the result
    st.write(f"Raw probability of cancer: {prediction_proba * 100:.2f}%")
    st.write(f"Adjusted prediction (threshold = {threshold}): {'HIGH' if prediction == 1 else 'LOW'}")