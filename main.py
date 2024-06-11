#import dependencies
import numpy as np
import pickle








#loading the saved model
loaded_model = pickle.load(open('D:/Users/Masterkim/projects/Diabetes/trained_model.sav', 'rb'))


#making a predictive system
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_array=np.asarray(input_data)
#reshape
input_array_reshape = input_array.reshape(1,-1)
prediction= loaded_model.predict(input_array_reshape)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')