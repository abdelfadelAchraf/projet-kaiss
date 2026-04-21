import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page
st.set_page_config(page_title="Energy Prediction App", page_icon="⚡", layout="wide")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("./src/models/best_model.pkl")
    scaler = joblib.load("./src/models/scaler.pkl")
    return model, scaler

@st.cache_data
def get_default_values():
    try:
        df = pd.read_csv("./src/data/buildingdata.csv")
        # Drop columns not used as features
        df = df.drop(columns=['Date', 'Id', 'Total electricity consumption'], errors='ignore')
        return df.mean().to_dict(), df.min().to_dict(), df.max().to_dict()
    except Exception as e:
        return {}, {}, {}

model, scaler = load_model()
defaults, mins, maxs = get_default_values()

# Title
st.title("⚡ Building Energy Consumption Prediction")
st.write("Adjust the parameters below to predict the **Total electricity consumption**. The default values are the means from the training dataset.")

st.sidebar.header("🔧 Building Features")
st.sidebar.write("Input parameters for prediction:")

# List of features used during training
features = [
    'Air Temperature', 'Radiant Temperature', 'Operative Temperature',
    'Outside Dry-Bulb Temperature', 'Glazing', 'Walls', 'Ceilings (int)',
    'Floors (int)', 'Ground Floors', 'Partitions (int)', 'Roofs',
    'External Infiltration', 'External Vent.', 'General Lighting',
    'Computer + Equip', 'Occupancy', 'Solar Gains Interior Windows',
    'Solar Gains Exterior Windows', 'Zone Sensible Heating',
    'Zone Sensible Cooling', 'Sensible Cooling', 'Total Cooling',
    'Mech Vent + Nat Vent + Infiltration'
]

# We will layout inputs in 4 columns on the main page instead of dropping all 23 in sidebar
cols = st.columns(4)
input_data = {}

for idx, feature in enumerate(features):
    col = cols[idx % 4]
    default_val = float(defaults.get(feature, 0.0))
    # We use number_input for precision
    input_data[feature] = col.number_input(feature, value=default_val, format="%.2f")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    predict_btn = st.button("🔮 Predict Energy Consumption", use_container_width=True)

with col2:
    if predict_btn:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale data
        try:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            st.success(f"## ⚡ Predicted Consumption: {prediction[0]:.2f} kWh/m²") # Or whatever the unit is
        except Exception as e:
            st.error(f"Error making prediction: {e}")

st.sidebar.write("---")
st.sidebar.info("This application uses the best performing machine learning model trained on the provided `buildingdata.csv` dataset.")
