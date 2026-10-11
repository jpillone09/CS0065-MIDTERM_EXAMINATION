import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
scaler = joblib.load("cafeteria_scaler.pkl")
model = joblib.load("cafeteria_kmeans.pkl")

st.title("Cafeteria Customer Pattern Grouping")

st.write(
    "Enter the customer's information to identify "
    "their customer group."
)

visit_frequency = st.number_input(
    "Visit Frequency per Month",
    min_value=1,
    max_value=100,
    value=5
)

average_spending = st.number_input(
    "Average Spending",
    min_value=1.0,
    max_value=5000.0,
    value=200.0
)

preferred_meal = st.selectbox(
    "Preferred Meal Category",
    [
        "Breakfast",
        "Lunch",
        "Dinner",
        "Snacks"
    ]
)

avg_visit_duration = st.number_input(
    "Average Visit Duration (minutes)",
    min_value=1,
    max_value=300,
    value=30
)

orders_per_visit = st.number_input(
    "Orders Per Visit",
    min_value=1,
    max_value=20,
    value=2
)

satisfaction_score = st.number_input(
    "Satisfaction Score",
    min_value=1.0,
    max_value=5.0,
    value=4.0
)

if st.button("Identify Customer Profile"):

    meal_data = {
        "Preferred_Meal_Breakfast":
            1 if preferred_meal == "Breakfast" else 0,

        "Preferred_Meal_Lunch":
            1 if preferred_meal == "Lunch" else 0,

        "Preferred_Meal_Dinner":
            1 if preferred_meal == "Dinner" else 0,

        "Preferred_Meal_Snacks":
            1 if preferred_meal == "Snacks" else 0
    }

    input_data = pd.DataFrame([{
        "Visit_Frequency": visit_frequency,
        "Average_Spending": average_spending,
        "Avg_Visit_Duration": avg_visit_duration,
        "Orders_Per_Visit": orders_per_visit,
        "Satisfaction_Score": satisfaction_score,
        **meal_data
    }])

    scaled_input = scaler.transform(input_data)

    cluster = model.predict(scaled_input)[0]

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )