import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

df = df[df['price']<3000000]
df = df[df['bedrooms']<10]
df = df[df['sqft_lot15'] < 400000]
df = df[df['sqft_lot'] < 600000]
df = df[df['sqft_living'] < 8000]
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date:date.year)
df['month'] = df['date'].apply(lambda date:date.month)
df.drop(columns=['id', 'zipcode', 'date'], inplace=True)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

X = scaler.fit_transform(df.drop('price', axis=1))
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Input(19,))

    for i in range(hp.Int('num_layers', min_value=1, max_value=8)):
        model.add(
            Dense(
                hp.Int('units'+str(i), min_value=8, max_value=128, step=8),
                activation=hp.Choice('activation'+str(i), values=['relu', 'tanh', 'sigmoid', 'linear'])
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(hp.Choice('droupout'+str(i), values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))

    model.add(Dense(1, activation='linear'))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['rmsprop', 'adam', 'sgd', 'nadam', 'adadelta']),
        loss='mse',
    )

    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    directory='tuners',
    project_name='house_prediction'
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model = tuner.get_best_models(num_models=1)[0]

callback = EarlyStopping(
    monitor="val_loss",
    min_delta=10000000,
    patience=50,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True
)

model.fit(X_train, y_train, epochs=2000, initial_epoch=5, batch_size=256 ,validation_data=(X_test, y_test), callbacks=callback)

predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMAE: ", mse)
print("MSE: ", mse)
print("RMSE: ", np.sqrt(mse))
print("\nR^2: ", r2)

from joblib import dump
dump(model, 'model.joblib') 
dump(scaler, 'scaler.joblib') 