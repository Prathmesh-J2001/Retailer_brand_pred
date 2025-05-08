import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("retailer_quantity_model.pkl")

# Load the feature dataset (Retailer Id as part of features)
@st.cache_data
def load_data():
    df = pd.read_csv("retailer_features.csv")
    return df

# Streamlit app layout
st.title("Retailer Brand-wise Purchase Quantity Predictor")

# Input Retailer ID
retailer_id = st.text_input("Enter Retailer ID:")

# Load the feature data
df = load_data()

# Define the list of brands for prediction (must match model output)
brands = ['AMUL', 'MACHO', 'SPORTO', 'SPORTO RED']

# Prediction block
if st.button("Predict Brand-wise Purchase Quantities"):
    if retailer_id and retailer_id.isdigit() and int(retailer_id) in df['Retailer Id'].values:
        # Keep Retailer Id in features if it was used during training
        retailer_features = df.loc[df['Retailer Id'] == int(retailer_id)]

        # Predict for each brand
        predictions = model.predict(retailer_features)[0]

        # Display predictions
        st.write(f"### Predicted Purchase Quantities for Retailer ID {retailer_id}:")
        for i, brand in enumerate(brands):
            st.success(f"{brand}: **{predictions[i]:.2f} units**")
    else:
        st.error("Retailer ID not found in the database or invalid input.")
