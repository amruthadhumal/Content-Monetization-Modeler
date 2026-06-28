import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Content Monetization Modeler",
    layout="wide",
    page_icon="💰"
)

st.markdown("""
    <style>
    body {background-color:#0e1117; color:#fff;}
    .sidebar .sidebar-content {background-color:#1c1c24;}
    h1, h2, h3, h4 { color: #F9FAFB !important; }
    .stNumberInput label, .stSelectbox label {
        color: #E0E0E0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Predict YouTube Ad Revenue")

MODEL_PATH = r"\Users\dhuma\OneDrive\Desktop\GUVI\Project_3\Current\linear_regression_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(" Model file not found. Please check the path.")
    st.stop()

model = joblib.load(MODEL_PATH)

feature_cols = [
    'video_duration_minutes',
    'views',
    'likes',
    'comments',
    'subscribers_gained',
    'avg_view_duration_seconds'
]

st.subheader("Enter Video Feature Values")

input_data = {}
cols_per_row = 3
rows = (len(feature_cols) + cols_per_row - 1) // cols_per_row

for i in range(rows):
    cols = st.columns(cols_per_row)
    for j in range(cols_per_row):
        idx = i * cols_per_row + j
        if idx < len(feature_cols):
            col_name = feature_cols[idx]
            input_data[col_name] = cols[j].number_input(f"{col_name}", value=0.0)


if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)  
        st.success(f"💰 **Predicted Ad Revenue:** ${prediction:,.2f}")

        if prediction == 0:
            st.info("This video may not generate ad revenue based on the current input values.")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")