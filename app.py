# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:01:11 2024

@author: Lenovo
"""

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

def main():
    st.title("CSV File Uploader")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(df)

        # Convert the DataFrame to a NumPy array
        df_array = np.array(df)

        # Load the trained Random Forest model
        model = load('C:/Users/Lenovo/Desktop/capstone project/random_forest_model.joblib"')
        
        # Make a prediction using the model
        prediction = model.predict(df_array)
        st.write("Predictions:")
        st.write(prediction)

if __name__ == "__main__":
    main()
