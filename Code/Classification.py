# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Funktionen für die Klassifikationsaufgabe
"""

# Bibliotheken importieren
from sklearn import metrics
import numpy as np


# Funktion für Erstellung des Modells
def train_model(clf, X_train, y_train, X_test):

    # Modell mit Daten trainieren
    clf.fit(X_train, np.ravel(y_train))

    # Testdaten vorhersagen
    y_pred = clf.predict(X_test)

    return clf, y_pred


# Funktion zum Testen des Modells
def score_model(y_pred, y_test):
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

    # Standardmetriken
    print('Recall:', recall)
    print('Precision:', precision)
    print('Accuracy:', accuracy)
    print('F1-Score:', f1_score)

    # Konfusionsmatrix
    print(metrics.confusion_matrix(y_test, y_pred))

    # Zusammenfassende Statistik
    print(metrics.classification_report(y_test, y_pred))
