import streamlit as st
import math
import pickle

# Load the machine learning model
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

# Set page title and header
st.title("Titanic Survival Prediction")
st.header("Get predictions based on passenger details")

# Define image URL for Titanic
titanic_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/St%C3%B6wer_Titanic.jpg/800px-St%C3%B6wer_Titanic.jpg"

# Display Titanic image
st.image(titanic_image_url, caption='RMS Titanic', use_column_width=True)

# Create columns for user inputs
col1, col2, col3 = st.columns(3)
with col1:
    Pclass = st.selectbox("Class of Passenger", ("Premiere", "Executive", "Economy"))
with col2:
    Sex = st.selectbox("Gender", ("Male", "Female")) 
with col3:
    Age = st.number_input("Age of passenger")

col4, col5 = st.columns(2)
with col4:
    SibSp = st.number_input("Siblings/Spouses")
with col5:
    Parch = st.number_input("Parents/Children")

col7, col8 = st.columns(2)
with col7:
    Fare = st.number_input("Fare of Journey")
with col8:
    Embarked = st.selectbox("Picking Point", ("Cherbourg", "Queenstown", "Southampton"))

# Predict survival on button click
if st.button("Predict"):
    pclass = 1 if Pclass == "Premiere" else 2 if Pclass == "Executive" else 3
    gender = 0 if Sex == "Male" else 1
    age = math.ceil(Age)
    sibsp = math.ceil(SibSp)
    parch = math.ceil(Parch)
    fare = math.ceil(Fare)
    embarked = 1 if Embarked == "Cherbourg" else 2 if Embarked == "Queenstown" else 3

    # Make prediction using the model
    result = model.predict([[pclass, gender, age, sibsp, parch, fare, embarked]])

    # Output prediction result
    output_labels = {1: "The passenger will Survive", 
                     0: "The passenger will not survive"}
    st.markdown(f"## Prediction: {output_labels[result[0]]}")
