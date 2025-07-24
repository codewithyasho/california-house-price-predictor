# app.py - Streamlit app using trained XGBoost Regressor

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Page config
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor with XGBoost")

# Input features
st.sidebar.header("Enter House Features")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.slider("Housing Median Age", 1, 100, 30)
total_rooms = st.sidebar.number_input("Total Rooms", value=880)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=129)
population = st.sidebar.number_input("Population", value=322)
households = st.sidebar.number_input("Households", value=126)
median_income = st.sidebar.number_input("Median Income", value=8.3252)
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# Collect input into DataFrame
input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# Load pre-trained model and pipeline


@st.cache_resource
def load_model():
    model = joblib.load("best_xgboost_model.pkl")
    pipeline = joblib.load("preprocessing_pipeline.pkl")
    return model, pipeline


model, pipeline = load_model()

# Prediction button
if st.button("Predict Price"):
    X_prepared = pipeline.transform(input_data)
    prediction = model.predict(X_prepared)
    st.success(f"🏡 Estimated House Price: ${prediction[0]:,.2f}")
