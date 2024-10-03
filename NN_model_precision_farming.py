# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:32:59 2023

@author: Giorgia
"""

import pandas as pd # provides high-performance, easy to use structures and data analysis tools
import numpy as np # provides fast mathematical computation on arrays and matrices
import tensorflow as tf # tensorflow and keras for building and training the neural network model (NN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler # to scale the NN input and ouput data 
from sklearn.model_selection import train_test_split # to subset the original dataset in training, validation and test part
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # metrics to measure the performance of the regression model
import matplotlib.pyplot as plt # to plot model loss and error functions
import time # to meaasure training time 
from sys import exit
import h5py

# load the dataset as pandas dataframe
data_train = pd.read_csv("data_train_path") #dataset in csv format

# show dataset properties
data_train  

'''
input and output columns difinition
'''
X_train = data_train.values[:,2:19]

y_train = data_train.values[:,19:20]

print(X_train.shape, y_train.shape)

'''
validation and testing subsets
'''
data_test = pd.read_csv("data_test_path") #dataset in csv format

# show dataset properties
data_test


X_val_and_test = data_test.values[:,2:19]

y_val_and_test = data_test.values[:,19:20]

X_train.shape, X_val_and_test.shape, y_train.shape, y_val_and_test.shape

y_val_and_test.shape

print('train',X_train)
print('train',y_train)

print(y_val_and_test)


'''
#Normalisation of input and output
'''

input_scaler = MinMaxScaler(feature_range=(-4, 4))
output_scaler = MinMaxScaler(feature_range=(-1, 1))

# with the input scaler we fit all the input space and then scale each splitted part of the dataset used as input for the neural network
input_scaler.fit(X_train)
X_train_scaled = input_scaler.transform(X_train)

X_val_and_test_scaled = input_scaler.transform(X_val_and_test)
# with the output scaler we fit all the output space and then scale each splitted part of the dataset used as output for the neural network
output_scaler.fit(y_train)
y_train_scaled = output_scaler.transform(y_train)


'''
Define and compile a sequential neural network with Keras
'''
model_nn_for_o3 = Sequential([Dense(units=10, 
                                    activation='tanh', 
                                    input_shape=(17,)),
                              Dense(units=3,
                                    activation='tanh'),
                              Dense(units=1,
                                    activation='linear')])

model_nn_for_o3.compile(optimizer='sgd',  
                        loss='mean_squared_error',  
                        metrics=['mae'])


# start of the training
start_training = time.time()

hist = model_nn_for_o3.fit(X_train_scaled, 
                           y_train_scaled, 
                           batch_size=10, 
                           epochs=300, 
                           validation_data=(X_val_and_test, y_val_and_test), 
                           verbose=1)

# measure the time needed for training the NN (with the current NN model configuration the time needed for training is about 60 seconds)
end_training = time.time()
print("Training time ", round((end_training - start_training), 3), "s" )


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# the final trained model makes prediction for the train, validation and test part of the dataset respectively
predict_train = model_nn_for_o3.predict(X_train_scaled)
predict_val_and_test = model_nn_for_o3.predict(X_val_and_test_scaled)
predict_train

# apply inverse transformation to have the same scale for estimated and true ozone concentration values for metrics computation
predict_train_inverse = output_scaler.inverse_transform(predict_train)
predict_val_inverse = output_scaler.inverse_transform(predict_val_and_test)
predict_train_inverse


# save the trained model
model_nn_for_o3.save("./model_nn_for_soil_nitrogen_estimation.h5")

###Evaluate the accuracy of the model predictions
mae_train = mean_absolute_error(y_train, predict_train_inverse)
mae_val = mean_absolute_error(y_val_and_test, predict_val_inverse)



print(f'Train MAE: {round(mae_train, 3)} , Val MAE: {round(mae_val, 3)} ')


rmse_train = mean_squared_error(y_train, predict_train_inverse, squared=True)
rmse_val = mean_squared_error(y_val_and_test, predict_val_inverse, squared=True)


#print(f'Train RMSE: {round(rmse_train, 3)} DU, Val RMSE: {round(rmse_val, 3)} DU, Test RMSE: {round(rmse_test, 3)} DU')
print(f'Train RMSE: {round(rmse_train, 3)}, Val RMSE: {round(rmse_val, 3)},')
pearson_train = np.sqrt(r2_score(y_train, predict_train_inverse,force_finite=(True)))
r2_val = (r2_score(y_val_and_test, predict_val_inverse,force_finite=(True)))
pearson_val = np.sqrt(r2_score(y_val_and_test, predict_val_inverse,force_finite=(True)))


print ('r2_val=',r2_val)
print(f'Train Pearson: {round(pearson_train, 3)}, Val Pearson: {round(pearson_val, 3)}')

#true_vs_predict plot
plt.plot([0.9,2.2],[0.9,2.2])
plt.scatter(y_val_and_test,predict_val_inverse)
plt.xlabel("TRUE_value (g/Kg)", size = 16,)
plt.ylabel("Predicted_Value(g/Kg)", size = 16)
plt.xscale("linear")
plt.yscale("linear")
plt.axis('square')
plt.xlim(0.9,2.2)
plt.ylim(0.9,2.2)
plt.show()