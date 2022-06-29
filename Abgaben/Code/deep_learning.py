# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Enthält Funktionen zur Erstellung von Deep-Learning-Modellen.
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import InputLayer, CategoryEncoding, Normalization
from autokeras import StructuredDataRegressor

import config as c


def find_optimal_model(X_train, y_train, X_test, y_test):
    """Optimale Netzwerkarchitektur mit autokeras bestimmen."""

    print("Search for optimal neural network started.")
    # define the search
    search = StructuredDataRegressor(max_trials=10, loss='mean_absolute_error')
    # perform the search
    search.fit(x=X_train, y=y_train, verbose=0)
    # evaluate the model
    mae, _ = search.evaluate(X_test, y_test, verbose=0)
    print('MAE: %.3f' % mae)
    # use the model to make a prediction
    X_new = np.asarray(X_test).astype('float32')
    yhat = search.predict(X_new)
    print('Predicted: %.3f' % yhat[0])
    # get the best performing model
    model = search.export_model()
    # summarize the loaded model
    print(model.summary())


def lstm_model():
    """Einfaches LSTM."""

    model = Sequential()
    model.add(LSTM(4, input_shape=([len(c.FEATURES), 1])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def lstm_model_2():
    """Erweitertes LSTM mit Dropout-Layern gegen Overfitting."""

    model = Sequential()
    model.add(LSTM(units=37, return_sequences=True,
                   input_shape=([len(c.FEATURES), 1]),
                   activation='tanh'))
    model.add(Dropout(0.2))

    model.add(LSTM(units=37, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.25))

    model.add(LSTM(units=37, activation='tanh'))
    model.add(Dropout(0.25))

    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def lstm_model_3():
    """Erweitertes LSTM mit Dropout-Layern gegen Overfitting mit
    anderem Optimizer."""

    model = Sequential()
    model.add(LSTM(units=37, return_sequences=True,
                   input_shape=([len(c.FEATURES), 1]),
                   activation='tanh'))
    model.add(Dropout(0.2))

    model.add(LSTM(units=37, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(LSTM(units=18, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(LSTM(units=10, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='SGD')

    return model


def keras_model():
    """Einfaches Feed-Forward-Netzwerk."""

    model = Sequential()
    model.add(Dense(units=37, input_dim=len(c.FEATURES),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(18, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def keras_model_2():
    """Erweitertes Feed-Forward-Netzwerk mit Dropout-Layern
    gegen Overfitting."""

    model = Sequential()
    model.add(Dense(units=37, input_dim=len(c.FEATURES),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(18, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def keras_model_3():
    """Erweitertes Feed-Forward-Netzwerk mit Dropout-Layern
    gegen Overfitting und SGD-Optimizer."""

    model = Sequential()
    model.add(Dense(units=37, input_dim=len(c.FEATURES),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(18, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='SGD')

    return model


def keras_model_4():
    """Erweitertes Feed-Forward-Netzwerk mit Dropout-Layern
    gegen Overfitting und Nadam-Optimizer."""

    model = Sequential()
    model.add(Dense(units=37, input_dim=len(c.FEATURES),
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(37, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(18, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='Nadam')

    return model


def auto_keras_model():
    """Erweitertes Feed-Forward-Netzwerk nach Vorschlag von autokeras."""

    model = Sequential()
    model.add(InputLayer(input_shape=(37,)))
    model.add(CategoryEncoding(num_tokens=37, output_mode="multi_hot"))
    model.add(Normalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model


def auto_keras_model_2():
    """Erweitertes Feed-Forward-Netzwerk nach Vorschlag von autokeras
    ohne den CategoryEncoding-Layer."""

    model = Sequential()
    model.add(InputLayer(input_shape=(37,)))
    # model.add(CategoryEncoding(num_tokens=37, output_mode="multi_hot"))
    model.add(Normalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    return model


def auto_keras_model_3():
    """Erweitertes Feed-Forward-Netzwerk nach Vorschlag von autokeras
    ohne die Layer CategoryEncoding und Normalization."""

    model = Sequential()
    model.add(InputLayer(input_shape=(37,)))
    # model.add(CategoryEncoding(num_tokens=37, output_mode="multi_hot"))
    # model.add(Normalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
