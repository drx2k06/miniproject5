import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")
st.write("Predict the estimated price of a house using Linear Regression.")

st.subheader("Enter House Details")

area = st.number_input(
    "Area (sq ft)",
    min_value=100,
    max_value=10000,
    value=1500,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

age = st.number_input(
    "Age of House (years)",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)

if st.button("Predict House Price"):
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, age]],
        columns=["area", "bedrooms", "bathrooms", "age"]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ₹{prediction:.2f} Lakhs")

st.markdown("---")
st.caption("House Price Prediction using Linear Regression")
