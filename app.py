import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import os

os.chdir(r'V:\DataScience\Main Projects\Deep Learning Projects\ANN_Bank')

#loading the train model
model = tf.keras.models.load_model('model.h5')

scaler=pickle.load(open("scaler.pkl","rb"))

gender_encoder=pickle.load(open("gender_encoder.pkl","rb"))

geo_encoder=pickle.load(open("Geopraphy_encoder.pkl","rb"))

# streamlit app:
st.title("Customer Churn Prediction")

# User Input:
creditscore=st.slider("Credit Score:",350,850,550)
geography=st.selectbox("Geography",['France', 'Spain', 'Germany'])
gender=st.selectbox("Gender",["Male","Female"])
age=st.slider("Age",18,90,40)
tenure=st.selectbox("Tenure",[1,2,3,4,5,6,7,8,9,10])
balance=st.number_input("Balance")
products=st.selectbox("Num Of Products",[1,2,3,4])
member=st.selectbox("Active/Not",["Active Member","Inactive Member"])
if member=="Active Member":
    member=0
else:
    member=1

# new input as dataframe
new_input={
    "CreditScore":[creditscore],
    "Geography":[geography],
    "Gender":[gender],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[products],
    "IsActiveMember":[member]

}

new_input=pd.DataFrame(new_input)

new_input["Geography"]=geo_encoder.transform(new_input["Geography"])
new_input["Gender"]=gender_encoder.transform(new_input["Gender"])

new_input=scaler.transform(new_input)



if st.button("Predict"):

    # Make prediction
    prediction=model.predict(new_input)
    pred=prediction[0][0]*100

    # Show prediction result
    st.subheader("Prediction Probability:")
    st.write(f"Chance of Customer Churn: {pred}%")