import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

df.drop(columns=['SkinThickness', 'BloodPressure'], inplace=True)
df = df.loc[(df['BMI'] < 50) | (df['Insulin'] < 100)]
df=df[df['Insulin'] < 500]
df=df[df['Pregnancies'] < 14]
df=df[df['DiabetesPedigreeFunction'] < 1.7]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

scaler = StandardScaler()

X = scaler.fit_transform(df.drop('Outcome', axis=1))
y = df['Outcome']

split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=101)
for train_index, test_index in split.split(X, y):
    strat_train_set = X[train_index], y.iloc[train_index]
    strat_test_set = X[test_index], y.iloc[test_index]

X_train, y_train = strat_train_set
X_test, y_test = strat_test_set

from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Input(6,))

    for i in range(hp.Int('num_layers', min_value=1, max_value=8)):
        model.add(
            Dense(
                hp.Int('units'+str(i), min_value=8, max_value=128, step=8),
                activation=hp.Choice('activation'+str(i), values=['relu', 'tanh', 'sigmoid'])
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(hp.Choice('droupout'+str(i), values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=hp.Choice('optimizer', values=['rmsprop', 'adam', 'sgd', 'nadam', 'adadelta']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='tuners',
    project_name='diabetes'
)

tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

model = tuner.get_best_models(num_models=1)[0]

callback = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.00001,
    patience=50,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True
)

model.fit(X_train, y_train, epochs=200, initial_epoch=5 ,validation_data=(X_test, y_test), callbacks=callback)

predictions = model.predict(X_test)
predictions = np.where(predictions > 0.45, 1,0)

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

from joblib import dump
dump(model, 'model.joblib') 
dump(scaler, 'scaler.joblib') 