import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("linear_regression_model.pkl")


# --------------------------------------------------
# Load Data & Model
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("youtube_ad_revenue_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("linear_regression_model.pkl")

df = load_data()
model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Project Overview",
        "Dataset",
        "Revenue Prediction"
    ]
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Project Overview":

    st.title("📺 YouTube Ad Revenue Predictor")

    st.write("""
    This application predicts YouTube advertising revenue
    based on video performance metrics.

    The prediction model was built using Machine Learning
    and trained on historical YouTube data.
    """)

    st.subheader("Project Features")

    st.markdown("""
    - View dataset information
    - Predict ad revenue
    - Interactive dashboard
    """)
# --------------------------------------------------
# Dataset
# --------------------------------------------------
elif page == "Dataset":

    st.title("📊 Dataset")

    st.write("Dataset Shape:", df.shape)

    st.dataframe(df.head())

    st.subheader("Column Information")

    st.write(df.dtypes)

# --------------------------
# User Inputs
# --------------------------
elif page == "Revenue Prediction":

    st.title("💰 Revenue Prediction")

    st.subheader("Enter video details")
    col1, col2 = st.columns(2)
    with col1:
       views = st.number_input("Views", min_value=0, value=10000)
       likes = st.number_input("Likes", min_value=0, value=500)
       comments = st.number_input("Comments", min_value=0, value=50)
       subscribers = st.number_input("Subscribers", min_value=0, value=5000)

    with col2:
       watch_time_minutes = st.number_input(
        "Watch Time (Minutes)",
        min_value=0.0,
        value=1000.0
       )

       video_length_minutes = st.number_input(
         "Video Length (Minutes)",
        min_value=0.0,
        value=10.0
       )

       year = st.number_input(
        "Year",
        min_value=2020,
        max_value=2035,
        value=2025
       )

       month = st.selectbox(
       "Month",
       list(range(1, 13))
       )

    day = st.selectbox(
    "Day",
    list(range(1, 32))
    )

    category = st.selectbox(
    "Category",
    [
        "Entertainment",
        "Gaming",
        "Lifestyle",
        "Music",
        "Tech"
    ]
    )

    device = st.selectbox(
    "Device",
    [
        "Mobile",
        "TV",
        "Tablet"
    ]
    )

    country = st.selectbox(
    "Country",
    [
        "CA",
        "DE",
        "IN",
        "UK",
        "US"
    ]
    )

    day_of_week = st.selectbox(
    "Day of Week",
    [
        0, 1, 2, 3, 4, 5, 6
    ]
    )
    

    #if st.button("Predict Revenue"):
    engagement_rate = ((likes + comments) / views if views > 0 else 0)

    # Base dataframe
    input_df = pd.DataFrame({
        "views": [views],
        "likes": [likes],
        "comments": [comments],
        "watch_time_minutes": [watch_time_minutes],
        "video_length_minutes": [video_length_minutes],
        "subscribers": [subscribers],
        "year": [year],
        "month": [month],
        "day": [day],
        "engagement_rate": [engagement_rate]
      })

    # Exact columns expected by model
    expected_columns = [
        'views',
        'likes',
        'comments',
        'watch_time_minutes',
        'video_length_minutes',
        'subscribers',
        'year',
        'month',
        'day',
        'category_Entertainment',
        'category_Gaming',
        'category_Lifestyle',
        'category_Music',
        'category_Tech',
        'device_Mobile',
        'device_TV',
        'device_Tablet',
        'country_CA',
        'country_DE',
        'country_IN',
        'country_UK',
        'country_US',
        'day_of_week_1',
        'day_of_week_2',
        'day_of_week_3',
        'day_of_week_4',
        'day_of_week_5',
        'day_of_week_6',
        'engagement_rate'
       ]

    # Add missing columns
    for col in expected_columns:
         if col not in input_df.columns:
            input_df[col] = 0

    # Set category
    input_df[f"category_{category}"] = 1

    # Set device
    input_df[f"device_{device}"] = 1

    # Set country
    input_df[f"country_{country}"] = 1

    # Set day_of_week
    if day_of_week != 0:
        input_df[f"day_of_week_{day_of_week}"] = 1

    # Ensure exact column order
    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted Ad Revenue: ${prediction:,.2f}"
    )

    #st.metric(
    #    "Predicted Revenue",
    #    f"${prediction:,.2f}"
    #)

    #st.write("Input Data")

    #st.dataframe(input_df)