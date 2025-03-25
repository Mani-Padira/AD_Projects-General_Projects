import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

df = pd.read_excel("restaurant_recommendation_dataset.xlsx")

X = df[['popularity_score', 'rating']]
y = df['delivery_time']
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
model.fit(X, y)

joblib.dump(model, "xgboost_model.pkl")

model = joblib.load("xgboost_model.pkl")

st.title("Food Delivery Time Predictor")

popularity_score = st.number_input("Enter Popularity Score", min_value=0.0, max_value=1.0, step=0.001, format="%.6f")
rating = st.number_input("Enter Rating", min_value=1.0, max_value=5.0, step=0.1)

if st.button("Predict Delivery Time"):
    prediction = model.predict([[popularity_score, rating]])[0]
    st.write(f"Predicted Delivery Time: {prediction:.2f} minutes")