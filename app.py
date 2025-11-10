import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    location_encoder = joblib.load('location_encoder.pkl')
    condition_encoder = joblib.load('condition_encoder.pkl')
    return model, scaler, location_encoder, condition_encoder

try:
    model, scaler, location_encoder, condition_encoder = load_model_and_preprocessors()
except FileNotFoundError as e:
    st.error(f"Model files not found. Please run the notebook to save the model first. Error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title and description
st.title("🏠 House Price Prediction")
st.markdown("Enter the details of the house to predict its price.")

# Create input form
with st.form("house_prediction_form"):
    st.header("House Details")
    
    # Numeric inputs
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (sq ft)", min_value=0, value=2000, step=100)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
    
    with col2:
        floors = st.number_input("Floors", min_value=1, max_value=5, value=2, step=1)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000, step=1)
    
    # Categorical inputs
    location = st.selectbox("Location", ["Downtown", "Rural", "Suburban", "Urban"])
    condition = st.selectbox("Condition", ["Excellent", "Fair", "Good", "Poor"])
    garage = st.selectbox("Garage", ["Yes", "No"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# Make prediction when form is submitted
if submitted:
    try:
        # Prepare numeric features
        numeric_features = np.array([[area, bedrooms, bathrooms, floors, year_built]])
        
        # Scale numeric features
        scaled_numeric = scaler.transform(numeric_features)
        
        # Encode location
        location_encoded = location_encoder.transform([[location]]).toarray()
        
        # Encode condition
        condition_encoded = condition_encoder.transform([[condition]]).toarray()
        
        # Encode garage (Yes: 1, No: 0)
        garage_encoded = np.array([[1 if garage == "Yes" else 0]])
        
        # Combine all features
        features = np.concatenate((scaled_numeric, location_encoded, condition_encoded, garage_encoded), axis=1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Display result
        st.success(f"### Predicted House Price: ${prediction:,.2f}")
        
        # Show feature importance or additional info
        with st.expander("View Details"):
            st.write(f"**Area:** {area:,} sq ft")
            st.write(f"**Bedrooms:** {bedrooms}")
            st.write(f"**Bathrooms:** {bathrooms}")
            st.write(f"**Floors:** {floors}")
            st.write(f"**Year Built:** {year_built}")
            st.write(f"**Location:** {location}")
            st.write(f"**Condition:** {condition}")
            st.write(f"**Garage:** {garage}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app predicts house prices using a machine learning model trained on historical house data.")
    st.write("Enter the house details in the form and click 'Predict Price' to get an estimate.")
    
    st.header("📊 Model Info")
    st.write("**Model Type:** Linear Regression")
    st.write("**Features Used:**")
    st.write("- Area, Bedrooms, Bathrooms")
    st.write("- Floors, Year Built")
    st.write("- Location, Condition, Garage")

