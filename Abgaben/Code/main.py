# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Hauptskript der semesterbegleitenden Aufgabe in ML in Manufacturing
"""

# Benötigte Bibliotheken importieren
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

import config as c
import data_input_and_feature_extraction as fe
import data_labeling as dl
import classification as cl
import regression as rg
import models
import deep_learning as deep


def create_data():
    """Einlesen der Messungen des FEMTO-Bearing-Datensatzes und
    extrahieren der Features."""

    # Trainingsdaten im Zeitbereich erstellen
    fe.import_export_time_domain(c.LEARNING_SETS, c.OUTPUT_TIME_DOMAIN,
                                 c.DATA_DIR, testset=False)

    # Testdaten im Zeitbereich erstellen
    fe.import_export_time_domain(c.TEST_SETS, c.OUTPUT_TIME_DOMAIN,
                                 c.DATA_DIR, testset=True)

    # Trainingsdaten im Frequenzbereich erstellen
    fe.import_export_frequency_domain(c.LEARNING_SETS,
                                      c.OUTPUT_FREQUENCY_DOMAIN,
                                      c.DATA_DIR, testset=False)

    # Testdaten im Frequenzbereich erstellen
    fe.import_export_frequency_domain(c.TEST_SETS,
                                      c.OUTPUT_FREQUENCY_DOMAIN,
                                      c.DATA_DIR, testset=True)

    # Statusnachricht ausgeben
    print("The Features were successfully extracted and saved!")


def label_data():
    """Eingelesenen Datensatz um Labelung für Klassifikation und
    Regression ergänzen."""

    # Trainingsdaten labeln und abspeichern
    train_data = dl.create_class_labeling(c.LEARNING_SETS,
                                          c.OUTPUT_DIR,
                                          testset=False)
    save_path = os.path.join(c.OUTPUT_LABELING, 'Lernset.csv')
    train_data.to_csv(save_path)

    # Testdaten labeln und abspeichern
    test_data = dl.create_class_labeling(c.TEST_SETS,
                                         c.OUTPUT_DIR,
                                         testset=True)
    save_path = os.path.join(c.OUTPUT_LABELING, 'Testset.csv')
    test_data.to_csv(save_path)


def prepare_classification_data():
    """Daten zur Klassifikation vorbereiten (Aufteilen in Trainings- und
    Testdaten, Bestimmung von Features und Zielgröße)"""

    # Trainingsdaten zur Klassifizierung einlesen
    filepath = os.path.join(c.INPUT_MODELS, 'Lernset.csv')
    df = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X_train = df[c.FEATURES_CLASSIFICATION]
    y_train = df[c.Y_CLASSIFICATION]

    # Merkmale skalieren
    X_train = StandardScaler().fit_transform(X_train)

    # Testdatensatz einlesen
    filepath = os.path.join(c.INPUT_MODELS, 'Testset.csv')
    df_test = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X_testset = df_test[c.FEATURES_CLASSIFICATION]
    y_testset = df_test[c.Y_CLASSIFICATION]

    # Merkmale skalieren
    X_testset = StandardScaler().fit_transform(X_testset)

    return X_train, y_train, X_testset, y_testset, df_test


def run_classification_models(X_train, y_train,
                              X_testset, y_testset, df_test):
    """Fitten der Klassifikationsmodelle und Bestimmen deren Genauigkeit
    (für das gesamte Testset und pro Bearing)"""

    # Laufvariable für Plots
    figure = 0

    # Modelle betrachten
    for name, model in models.TUNED_CLASSIFIER.items():
        print('=' * 20)
        print('Modell:', name)

        # Modell trainieren
        clf, y_pred = cl.train_model(model, X_train, y_train, X_train)

        # Genauigkeit für Testdaten aus Trainingsdaten bestimmen
        print('Ergebnisse des Trainings:')
        cl.score_model(y_pred, y_train)

        # Feature Importance
        # Quelle: https://campus.datacamp.com/courses/
        # machine-learning-with-tree-based-models-in-python/
        # bagging-and-random-forests?ex=10
        # Create a pd.Series of features importances
        try:
            importances = pd.Series(data=clf.feature_importances_,
                                    index=c.FEATURES_CLASSIFICATION)

            # Sort importances
            importances_sorted = importances.sort_values()

            # Draw a horizontal barplot of importances_sorted
            plt.figure(figure)
            importances_sorted.plot(kind='barh', color='lightgreen')
            plt.title('Features Importances for model {}'.format(name))
            plt.show()
            figure += 1
        except AttributeError:
            pass

        # Hyperparamter-Tuning
        # cl.tune_classifier(model=model, params=models.HYPERPARAMS,
        #                    X_train=X_train, y_train=y_train)

        # Genauigkeit für Testdatensatz bestimmen
        print('Ergebnisse des gesamten Testsets:')
        predictions = clf.predict(X_testset)
        cl.score_model(predictions, y_testset)

        # ROC-Kurven für jeweils 2 Klassen (0 & 1, 1 & 2, 0 & 2) ausgeben
        plt.figure(figure)
        cl.show_roc_curve(y_testset, predictions,
                          title=f'ROC-Curve for model {name}')
        figure += 1

        # Betrachtung der einzelnen Bearings
        for bearing in c.TEST_SETS:
            print('Ergebnisse für {}:'.format(bearing))
            reduced_set = df_test.loc[bearing, slice(None), :]
            reduced_X = reduced_set[c.FEATURES_CLASSIFICATION]
            reduced_y = reduced_set[c.Y_CLASSIFICATION]

            # Merkmale skalieren
            reduced_X = StandardScaler().fit_transform(reduced_X)

            predictions = clf.predict(reduced_X)
            cl.score_model(predictions, reduced_y)

            # Vorhergesagte RUL-Klasse mit tatsächlicher Labelung vergleichen
            plt.figure(figure)
            plt.plot(reduced_y['RUL_Class'].to_list(), color='black',
                     label='Real', alpha=0.5)
            plt.plot(predictions, color='green', label='Predictions',
                     alpha=0.5, linestyle='none', marker='x')
            plt.title('Predicted RUL-Class for {} model {}'.format(bearing,
                                                                   name))
            plt.xlabel('TIme')
            plt.ylabel('RUL-Class')
            plt.legend()
            plt.show()
            figure += 1


def prepare_regression_data():
    """Daten zur Regression vorbereiten (Aufteilen in Trainings- und
    Testdaten, Bestimmung von Features und Zielgröße)"""

    # Trainingsdaten zur Regression einlesen
    filepath = os.path.join(c.INPUT_MODELS, 'Lernset.csv')
    df = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X_train = df[c.FEATURES]
    y_train = df[c.Y_REGRESSION]

    # Merkmale skalieren
    X_train = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_train)

    # Testdatensatz einlesen
    filepath = os.path.join(c.INPUT_MODELS, 'Testset.csv')
    df_test = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X_testset = df_test[c.FEATURES]
    y_testset = df_test[c.Y_REGRESSION]

    # Merkmale skalieren
    X_testset = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_testset)

    return X_train, y_train, X_testset, y_testset, df_test


def run_regression_models(X_train, y_train,
                          X_testset, y_testset, df_test):
    """Fitten der Regressionsmodelle und Bestimmen deren Genauigkeit
    (für das gesamte Testset und pro Bearing)"""

    figure = 1

    # Modelle betrachten
    for name, model in models.REGRESSORS.items():
        print('=' * 20)
        print('Modell:', name)

        # Modell trainieren
        reg, y_pred = rg.train_model(model, X_train, y_train)

        # Genauigkeit für Testdaten aus Trainingsdaten bestimmen
        print('Ergebnisse des Trainings:')
        rg.score_model(y_pred, y_train)

        # Genauigkeit für Testdatensatz bestimmen
        print('\nErgebnisse des gesamten Testsets:')
        predictions = reg.predict(X_testset)
        rg.score_model(predictions, y_testset)

        # Betrachtung der einzelnen Bearings
        for bearing in c.TEST_SETS:
            print('\nErgebnisse für {}:'.format(bearing))
            reduced_set = df_test.loc[bearing, slice(None), :]
            reduced_X = reduced_set[c.FEATURES]
            reduced_y = reduced_set[c.Y_REGRESSION]

            # Merkmale skalieren
            reduced_X = MinMaxScaler(feature_range=(0, 1)) \
                .fit_transform(reduced_X)

            predictions = reg.predict(reduced_X)
            rg.score_model(predictions, reduced_y)

            plt.figure(figure)
            plt.plot(reduced_y['RUL'].to_list(), color='black', label='Real')
            plt.plot(predictions, color='green', label='Predictions')
            plt.title('RUL Prediction for {} model {}'.format(bearing, name))
            plt.xlabel('Observation')
            plt.ylabel('RUL')
            plt.legend()
            plt.show()
            figure += 1


def run_deep_learning_model(model, epochs, batch_size, X_train, y_train,
                            X_test, y_test, df_test):
    """Fitten des Deep-Learning-Modells und Bestimmen dessen Genauigkeit
    (für das gesamte Testset und pro Bearing)"""

    training_hist = model.fit(X_train, y_train, epochs=epochs,
                              batch_size=batch_size,
                              verbose=1,
                              validation_data=(X_test, y_test))
    # make predictions
    trainPredict = model.predict(X_train)

    print('Ergebnisse des Trainings:')
    rg.score_model(trainPredict, y_train)

    # Genauigkeit für Testdatensatz bestimmen
    print('\nErgebnisse des gesamten Testsets:')
    predictions = model.predict(X_test)
    rg.score_model(predictions, y_test)

    # Early Stopping Kurve
    plt.figure(1)
    x_cords = np.arange(epochs)+1
    loss = training_hist.history['loss']
    val_loss = training_hist.history['val_loss']
    plt.plot(x_cords, loss, label='Trainig-Loss')
    plt.plot(x_cords, val_loss, label='Validierung-Loss')
    plt.title('Vergleich von Training und Validierungs Loss-Funktion')
    plt.xlabel('Epochen')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

    figure = 2

    # Testsets einzeln betrachten
    for bearing in c.TEST_SETS:
        print('\nErgebnisse für {}:'.format(bearing))
        reduced_set = df_test.loc[bearing, slice(None), :]
        reduced_X = reduced_set[c.FEATURES]
        reduced_y = reduced_set[c.Y_REGRESSION]

        # Merkmale skalieren
        reduced_X = MinMaxScaler(feature_range=(0, 1)).fit_transform(reduced_X)

        predictions = model.predict(reduced_X)
        rg.score_model(predictions, reduced_y)

        plt.figure(figure)
        plt.plot(reduced_y['RUL'].to_list(), color='black', label='Real')
        plt.plot(predictions, color='green', label='Predictions')
        plt.title('RUL Prediction for {}'.format(bearing))
        plt.xlabel('Time')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()
        figure += 1


def main():
    """Ausführen des Projekts."""

    # Warnungen unterdrücken
    warnings.filterwarnings('ignore')

    # Test- und Trainingsdaten erzeugen
    # create_data()

    # Daten labeln
    # label_data()

    # Daten zur Klassifikation vorbereiten
    X_train, y_train, X_testset, y_testset, df_test = \
        prepare_classification_data()

    # Klassifikation durchführen
    run_classification_models(X_train, y_train, X_testset, y_testset,
                              df_test)

    # Daten zur Regression vorbereiten
    X_train, y_train, X_testset, y_testset, df_test = \
        prepare_regression_data()

    # Regression durchführen
    run_regression_models(X_train, y_train, X_testset, y_testset, df_test)

    # Daten für Neuronales Netz shuffeln
    X_train, y_train = shuffle(X_train, y_train, random_state=22)

    # Optimales Deep-Learning Modell mit Autokeras finden
    # deep.find_optimal_model(X_train, y_train, X_testset, y_testset)

    # Deep-Learning
    run_deep_learning_model(deep.keras_model_3(), epochs=50,
                            batch_size=16, X_train=X_train, y_train=y_train,
                            X_test=X_testset, y_test=y_testset,
                            df_test=df_test)


if __name__ == '__main__':
    main()
