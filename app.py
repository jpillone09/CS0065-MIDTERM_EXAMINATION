
import streamlit as st
import pandas as pd
import joblib

scaler = joblib.load("cafeteria_scaler.pkl")
model = joblib.load("cafeteria_kmeans.pkl")

st.title("Cafeteria Customer Pattern Grouping")
st.write("Enter the customer's information to identify their customer group.")

# Use a form to keep inputs together
with st.form("customer_form"):

    visit_frequency = st.number_input(
        "Visit Frequency per Month",
        min_value=1,
        max_value=100,
        value=None,
        placeholder="Enter 1–100"
    )

    average_spending = st.number_input(
        "Average Spending",
        min_value=1.0,
        max_value=5000.0,
        value=None,
        placeholder="Enter 1–5000"
    )

    preferred_meal = st.selectbox(
        "Preferred Meal Category",
        ["Select a meal", "Breakfast", "Lunch", "Dinner", "Snacks"]
    )

    avg_visit_duration = st.number_input(
        "Average Visit Duration (minutes)",
        min_value=1,
        max_value=300,
        value=None,
        placeholder="Enter 1–300"
    )

    orders_per_visit = st.number_input(
        "Orders Per Visit",
        min_value=1,
        max_value=20,
        value=None,
        placeholder="Enter 1–20"
    )

    satisfaction_score = st.number_input(
        "Satisfaction Score",
        min_value=1.0,
        max_value=5.0,
        value=None,
        placeholder="Enter 1–5"
    )

    submitted = st.form_submit_button("Identify Customer Profile")

if submitted:
    # Check for missing inputs
    values = [
        visit_frequency,
        average_spending,
        avg_visit_duration,
        orders_per_visit,
        satisfaction_score
    ]

    if any(v is None for v in values) or preferred_meal == "Select a meal":
        st.error("Please complete all fields before identifying the customer profile.")

    else:
        meal_data = {
            "Preferred_Meal_Breakfast": int(preferred_meal == "Breakfast"),
            "Preferred_Meal_Lunch": int(preferred_meal == "Lunch"),
            "Preferred_Meal_Dinner": int(preferred_meal == "Dinner"),
            "Preferred_Meal_Snacks": int(preferred_meal == "Snacks")
        }

        input_data = pd.DataFrame([{
            "Visit_Frequency": visit_frequency,
            "Average_Spending": average_spending,
            "Avg_Visit_Duration": avg_visit_duration,
            "Orders_Per_Visit": orders_per_visit,
            "Satisfaction_Score": satisfaction_score,
            **meal_data
        }])

        try:
            scaled_input = scaler.transform(input_data)
            cluster = model.predict(scaled_input)[0]

            st.success(f"Customer belongs to Cluster {cluster}")
            st.write(
                "This customer has been grouped according to "
                "similarities in their entered characteristics."
            )

        except Exception as e:
            st.error(
                "Could not process these inputs. Check that the "
                "features and their order match the trained model."
            )
            st.caption(str(e))
