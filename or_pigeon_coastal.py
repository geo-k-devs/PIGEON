# OR_PIGEON_coastal.py

# Libraries Import

# check that all the requested python modules are properly installed
try:

    import geopandas as gpd
    import pandas as pd # provides high-performance, easy to use structures and data analysis tools
    import numpy as np # provides fast mathematical computation on arrays and matrices
    import json
    import tensorflow as tf # tensorflow and keras for building and training the neural network model (NN)
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler # to scale the NN input and ouput data
    from sklearn.model_selection import train_test_split # to subset the original dataset in training, validation and test part
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # metrics to measure the performance of the regression model
    import matplotlib.pyplot as plt # to plot model loss and error functions
    import time # to meaasure training time
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    import optuna
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Dropout, Flatten
    from keras.optimizers import Adam,RMSprop
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import pickle
    import csv

except ModuleNotFoundError:
    print("Module import error")
else:
    print('\nAll libreries properly installed. Ready to start!', '\n')

# COASTAL

# load the dataset as pandas dataframe
dataset_coastal = pd.read_csv("./DATASET_OR_coastal.csv", sep=';')

dataset_coastal

X = dataset_coastal.values[:, 2:12]
y = dataset_coastal.values[:, 12:13]
X.shape, y.shape

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X,
                                                                    y,
                                                                    test_size=0.4,
                                                                    random_state=32)

X_train.shape, X_val_and_test.shape, y_train.shape, y_val_and_test.shape

# use part of the dataset for validation (55%) and part for testing the neural network (45%)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test,
                                                y_val_and_test,
                                                test_size=0.45,
                                                random_state=32)

X_val.shape, X_test.shape, y_val.shape, y_test.shape

input_scaler = MinMaxScaler(feature_range=(-5, 5))
output_scaler = MinMaxScaler(feature_range=(0, 1))

# with the input scaler we fit all the input space and then scale each splitted part of the dataset used as input for the neural network
input_scaler.fit(X)
X_train_scaled = input_scaler.transform(X_train)
X_val_scaled = input_scaler.transform(X_val)
X_test_scaled = input_scaler.transform(X_test)

X_train_scaled

# with the output scaler we fit all the output space and then scale each splitted part of the dataset used as output for the neural network
output_scaler.fit(y)
y_train_scaled = output_scaler.transform(y_train)
y_val_scaled = output_scaler.transform(y_val)

#optuna tuning

output_name = f'best_trial_param.pkl'

def objective(trial):
  model_nn = Sequential([
        Flatten(),
        Dense(trial.suggest_int("dense_units_1", 16, 64), activation=trial.suggest_categorical('af1',['relu','sigmoid']), input_shape=(10,)),
        Dense(trial.suggest_int("dense_units_2", 8, 32), activation=trial.suggest_categorical('af2',['relu','sigmoid'])),
        Dropout(trial.suggest_float("dropout", 0.0, 0.5)),
        Dense(1, activation=trial.suggest_categorical('af3',['relu','sigmoid','softmax']))])

  model_nn.compile(optimizer=Adam(learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)),
                  loss='mean_squared_error',
                  metrics=['mae'])

  early_stopping = EarlyStopping(monitor='val_loss', patience=trial.suggest_int("early_stopping_patience", 10, 30),
                                   restore_best_weights=True, verbose=1)

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5,factor=trial.suggest_float("reduce_lr", 0.01, 0.9),min_lr=0.00001, verbose=0)

  history = model_nn.fit(X_train_scaled, y_train_scaled,
                        batch_size=32,
                        epochs=1000,
                        validation_data=(X_val_scaled, y_val_scaled),
                        callbacks=[early_stopping, reduce_lr],
                        verbose=0)

  return history.history['val_loss'][-1]  # return the final validation loss

# After optimizing with Optuna

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")  # Note: Optuna minimizes the objective function
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")

    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best trial object to a file
    with open(output_name, 'wb') as f:
        pickle.dump(best_trial, f)

with open(f'best_trial_param.pkl', 'rb') as f: best_trial = pickle.load(f)
print("Best trial:")
print("  Value: {}".format(best_trial.value))
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

model_nn = Sequential([
        Flatten(),
        Dense(units=best_trial.params['dense_units_1'], activation=best_trial.params['af1'], input_shape=X.shape),
        Dense(units=best_trial.params['dense_units_2'], activation=best_trial.params['af2']),
        Dropout(best_trial.params['dropout']),
        Dense(1, activation=best_trial.params['af3'])
])

optimizer = Adam(learning_rate=best_trial.params['learning_rate'])
model_nn.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5,factor=best_trial.params["reduce_lr"],min_lr=0.00001, verbose=1)

# start of the training
start_training = time.time()

model_chl_coastal = model_nn

hist = model_chl_coastal.fit(X_train_scaled,
                            y_train_scaled,
                            batch_size=5,
                            epochs=1000,
                            validation_data=(X_val_scaled, y_val_scaled),
                            verbose=1)

# measure the time needed for training the NN
end_training = time.time()
print("Training time ", round((end_training - start_training)/60, 3), "min" )

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
predict_train = model_chl_coastal.predict(X_train_scaled)
predict_val = model_chl_coastal.predict(X_val_scaled)
predict_test = model_chl_coastal.predict(X_test_scaled)

# apply inverse transformation to have the same scale for estimated and true chl concentration values for metrics computation
predict_train_inverse = output_scaler.inverse_transform(predict_train)
predict_val_inverse = output_scaler.inverse_transform(predict_val)
predict_test_inverse = output_scaler.inverse_transform(predict_test)

# save the trained model
model_chl_coastal.save("./nn_OR_coastal.h5")

mae_train = mean_absolute_error(y_train, predict_train_inverse)
mae_val = mean_absolute_error(y_val, predict_val_inverse)
mae_test = mean_absolute_error(y_test, predict_test_inverse)
print(f'Train MAE: {round(mae_train, 3)} mg/m3, Val MAE: {round(mae_val, 3)} mg/m3, Test MAE: {round(mae_test, 3)} mg/m3')

rmse_train = mean_squared_error(y_train, predict_train_inverse, squared=False)
rmse_val = mean_squared_error(y_val, predict_val_inverse, squared=False)
rmse_test = mean_squared_error(y_test, predict_test_inverse, squared=False)
print(f'Train RMSE: {round(rmse_train, 3)} mg/m3, Val RMSE: {round(rmse_val, 3)} mg/m3, Test RMSE: {round(rmse_test, 3)} mg/m3')

r2_train = r2_score(y_train, predict_train_inverse)
r2_val = r2_score(y_val, predict_val_inverse)
r2_test = r2_score(y_test, predict_test_inverse)
print(f'Train R\u00b2: {round(r2_train, 3)}, Val R\u00b2: {round(r2_val, 3)}, Test R\u00b2: {round(r2_test, 3)}')

pearson_train = np.sqrt(r2_score(y_train, predict_train_inverse))
pearson_val = np.sqrt(r2_score(y_val, predict_val_inverse))
pearson_test = np.sqrt(r2_score(y_test, predict_test_inverse))
print(f'Train Pearson: {round(pearson_train, 3)}, Val Pearson: {round(pearson_val, 3)}, Test Pearson: {round(pearson_test, 3)}')

print("\n" "Evaluation of mean and standard deviation of the estimated values of CHL compared to the actual ones")
print(f'CHL true mean: {round(np.mean(y_test), 3)} mg/m3 --- CHL estimated mean: {round(float(np.mean(predict_test_inverse)), 3)} mg/m3')
print(f'CHL true std: {round(np.std(y_test), 3)} mg/m3 --- CHL estimated std: {round(float(np.std(predict_test_inverse)), 3)} mg/m3')

plt.plot([0,5],[0,5])
plt.scatter(y_test, predict_test_inverse, c='b', s=10)
plt.scatter(y_train, predict_train_inverse, c='b', s=10)
plt.scatter(y_val, predict_val_inverse, c='b', s=10)
plt.xlabel("Chl-a Concentration (mg/m³)", size = 10,)
plt.ylabel("Estimated Chl-a Concentration (mg/m³)", size = 10)
plt.title('Estimation OR Coastal')
plt.xscale("linear")
plt.yscale("linear")
plt.axis('square')
plt.tight_layout()
plt.xlim(0,5)
plt.ylim(0,5)
plt.show()

model_chl_coastal.summary()