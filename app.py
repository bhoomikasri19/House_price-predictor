import streamlit as st
import joblib
import numpy as np
import pandas as pd

#load model
model=joblib.load("model/houseprice.pkl")
#load columns
columns=joblib.load("model/columns.pkl")

st.title("House price predictor")

#input

area=st.number_input("Carpet Area")
floor=st.number_input("Floor")
bathroom=st.number_input("Bathroom")
balcony=st.number_input("Balcony")
locations = [col.replace('location_', '') for col in columns if 'location_' in col]
locations.sort()
location = st.selectbox("Location", locations)

if st.button("Predict"):
    

    input_dict=dict.fromkeys(columns,0)
    input_dict['Carpet Area']=area
    input_dict['Floor']=floor
    input_dict['Bathroom']=bathroom
    input_dict['Balcony']=balcony

    loc_col='location_'+location.lower()
    if loc_col in columns:
        input_dict[loc_col]=1

    input_df=pd.DataFrame([input_dict])
    input_df=input_df[columns]


    #Prediction
    prediction=model.predict(input_df)[0]

    st.success(f"Prediction Price:{round(prediction,2)}Lakhs")