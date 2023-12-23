import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

loaded_model=pickle.load(open('medical_Insurance.sav','rb'))

def medical_cost_pred(input_data):

    # changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return 'The insurance cost is USD: ', prediction[0]

def main():
    st.title('Medical Insurance Cost Prediction')

    age=st.number_input('Enter the age')
    sex=st.number_input('Enter the sex')
    bmi=st.number_input('Enter the BMI')
    children=st.number_input('Enter the number of children')
    smoker=st.number_input('Smoker or not')
    region=st.number_input('Enter the region')


    diagnosis=''
    if st.button('Insurance_Cost'):
        diagnosis=medical_cost_pred([age,sex,bmi,children,smoker,region])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
