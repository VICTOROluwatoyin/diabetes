import numpy as np
import pickle
import streamlit as st








#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


#create a function for prediction

def diabetes_pred(input_data):
    #changing the input data to numpy array

    input_array=np.asarray(input_data)

    #reshape

    input_array_reshape = input_array.reshape(1,-1)
    prediction= loaded_model.predict(input_array_reshape)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    

def main():

    #giving a title
    st.title("Diabetes web app")

    #getting input data from users

    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose value')
    BloodPressure = st.text_input('Systolic blood pressure value')
    SkinThickness = st.text_input('Skin thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('Body mass index')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value')
    Age = st.text_input('Age of the patient')


    #code for prediction
    diagnosis = ' '
    
    #creating a button for prediction

    if st.button('Check result'):
        diagnosis = diabetes_pred([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ])

    st.success(diagnosis)







if __name__ == "__main__":
    main()



















