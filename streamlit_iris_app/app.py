import streamlit as st
import numpy as np
import pickle
import os

model_path = "iris_model.pkl"

if not os.path.exists(model_path):
    st.error("🚫 Model file not found. Make sure 'iris_model.pkl' is in the repo.")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🌸 Iris Flower Predictor")
st.write("Enter measurements below:")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Iris Species: **{species[prediction]}**")
