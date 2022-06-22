# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 12:01:34 2022

@author: 1
"""

import numpy as np

from sklearn import metrics
# from sklearn.metrics import explained_variance_score
# from sklearn.metrics import max_error


# Funktion für Erstellung des Modells
def train_model(reg, X_train, y_train, X_test):

    # Modell mit Daten trainieren
    reg.fit(X_train, np.ravel(y_train))

    # Testdaten vorhersagen
    y_pred = reg.predict(X_test)

    return reg, y_pred


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Funktion zum Testen des Modells
def score_model(y_pred, y_test):
    explained_variance = metrics.explained_variance_score(y_test, y_pred)
    max_error = metrics.max_error(y_test, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error = mean_squared_error ** 0.5
    r2_score = metrics.r2_score(y_test, y_pred)
    mean_absolute_percentage_error = \
        metrics.mean_absolute_percentage_error(y_test, y_pred)
    # mape = mean_absolute_percentage_error(y_test, y_pred)

    # Standardmetriken
    print('Explained Variance Score:', explained_variance)
    print('Max Error:', max_error)
    print('Mean Absolute Error:', mean_absolute_error)
    print('Mean Squared Error:', mean_squared_error)
    print('Root Mean Squared Error:', root_mean_squared_error)
    print('R2-Score:', r2_score)
    print('Mean Absolute Percentage Error:', mean_absolute_percentage_error)
