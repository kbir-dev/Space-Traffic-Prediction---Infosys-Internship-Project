import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load pre-trained models
models = {
    "Linear Regression": "linear_regression_model.pkl",
    "Ridge Regression": "best_ridge_model.pkl",
    "Lasso Regression": "best_lasso_model.pkl",
    "KNN": "best_knn_model.pkl",
    "Decision Tree": "best_decision_tree_model.pkl",
    "Random Forest": "best_random_forest_model.pkl"
}

# Preprocessing pipeline
def preprocess_input(data, preprocessor):
    """
    Preprocess input data using the preprocessor.
    """
    X = data[['Location', 'Object_Type', 'Hour', 'Day_of_Week', 'Day_of_Month']]
    X_processed = preprocessor.transform(X)
    return X_processed

# Load the preprocessing pipeline
@st.cache_resource
def load_preprocessor():
    """
    Create and return the preprocessor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('location_object', OneHotEncoder(), ['Location', 'Object_Type']),
            ('hour', 'passthrough', ['Hour', 'Day_of_Week', 'Day_of_Month'])
        ]
    )
    return preprocessor

# Load model function
@st.cache_resource
def load_model(model_path):
    """
    Load and return the machine learning model from the given path.
    """
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

# Load scaler function
@st.cache_resource
def load_scaler(scaler_path):
    """
    Load and return the StandardScaler used during model training.
    """
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
    return scaler

# Title and Introduction
st.title("Space Traffic Density Prediction")
st.write("This application predicts traffic density in space based on input features like location, object type, and time.")

# Inputs
locations = ["Lagrange Point L2", "Orbit LEO", "Mars Transfer Orbit", "Lagrange Point L1", "Orbit GEO", "Orbit MEO"]
object_types = ["Space Station", "Satellite", "Scientific Probe", "Space Debris", "Manned Spacecraft", "Asteroid Mining Ship"]

# Input fields
selected_date = st.date_input("Select Date")
selected_time = st.time_input("Select Time")
selected_location = st.selectbox("Select Location", locations)
selected_object_type = st.selectbox("Select Object Type", object_types)
selected_model = st.selectbox("Choose a Model", list(models.keys()))

# Process inputs
if st.button("Predict"):
    # Combine date and time into a single datetime
    selected_datetime = pd.to_datetime(f"{selected_date} {selected_time}")

    # Extract features from datetime
    hour = selected_datetime.hour
    day_of_week = selected_datetime.weekday()
    day_of_month = selected_datetime.day

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Location": [selected_location],
        "Object_Type": [selected_object_type],
        "Hour": [hour],
        "Day_of_Week": [day_of_week],
        "Day_of_Month": [day_of_month]
    })

    # Load preprocessor, model, and scaler
    preprocessor = load_preprocessor()
    model = load_model(models[selected_model])
    scaler = load_scaler('y_scaler.pkl')  # Path to your saved scaler (used during model training)

    # Fit the preprocessor to the categories (if needed)
    preprocessor.fit(input_data)  # You can adjust this based on your dataset preprocessing pipeline
    
    # Preprocess inputs
    X_processed = preprocess_input(input_data, preprocessor)

    # Predict
    prediction = model.predict(X_processed)

    # Rescale the prediction
    rescaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

    # Display results
    st.write(f"Predicted Space Traffic Density: {float(rescaled_prediction[0]):.2f}")
