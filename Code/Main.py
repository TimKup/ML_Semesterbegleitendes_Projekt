# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler
"""


# Benötigte Bibliotheken importieren
import Config as c
import Data_Input_and_Feature_Extraction as fe
import Data_Labeling as dl
import Classification as cl
import Helper_Functions as helper
import Regression as rg

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA

import warnings
import math

from sklearn.metrics import mean_squared_error


def create_data():
    # Trainingsdaten im Zeitbereich erstellen
    fe.import_export_time_domain(c.LEARNING_SETS, c.OUTPUT_TIME_DOMAIN,
                                 c.DATA_DIR, testset=False)

    # Testdaten im Zeitbereich erstellen
    fe.import_export_time_domain(c.TEST_SETS, c.OUTPUT_TIME_DOMAIN,
                                 c.DATA_DIR, testset=True)

    # Trainingsdaten im Frequenzbereich erstellen
    fe.import_export_frequency_domain(c.LEARNING_SETS,
                                      c.OUTPUT_FREQUENCY_DOMAIN,
                                      c.DATA_DIR,  testset=False)

    # Testdaten im Frequenzbereich erstellen
    fe.import_export_frequency_domain(c.TEST_SETS,
                                      c.OUTPUT_FREQUENCY_DOMAIN,
                                      c.DATA_DIR,  testset=True)


def label_data():
    # Trainingsdaten labeln und abspeichern
    train_data = dl.create_class_labeling(c.LEARNING_SETS,
                                          c.OUTPUT_DIR,
                                          testset=False)
    save_path = os.path.join(c.OUTPUT_LABELING, 'Lernset_Klassifikation.csv')
    train_data.to_csv(save_path)

    # Testdaten labeln und abspeichern
    test_data = dl.create_class_labeling(c.TEST_SETS,
                                         c.OUTPUT_DIR,
                                         testset=True)
    save_path = os.path.join(c.OUTPUT_LABELING, 'Testset_Klassifikation.csv')
    test_data.to_csv(save_path)

    # TODO: Entfernen?
    # # Labels anzeigen
    # file = os.path.join(c.INPUT_LABELING, 'Lernsets',
    #                     'Bearing1_1_time_params.csv')
    # title = 'Bearing 1_1 - Grenzwert: 2g - Klassengrenze: 1.225 h - ' \
    #         'ohne Annahme der Zerstörung am Ende'
    # helper.visualize_labels(file, title=title)


def prepare_classification_data():
    # Trainingsdaten zur Klassifizierung einlesen
    filepath = os.path.join(c.INPUT_CLASSIFIER, 'Lernset_Klassifikation.csv')
    df = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X = df[c.FEATURES]
    y = df[c.Y_CLASSIFICATION]

    # Merkmale skalieren
    X = StandardScaler().fit_transform(X)

    # Dimensionsreduzierung
    # pca = PCA(n_components=0.95)
    # X = pca.fit_transform(X)
    # global n_components
    # n_components = len(X[0])
    # print(pca.explained_variance_ratio_)

    # Train-Test-Split - ggf. stratify=y?
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        random_state=14)

    # Testdatensatz einlesen
    filepath = os.path.join(c.INPUT_CLASSIFIER, 'Testset_Klassifikation.csv')
    df_test = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X_testset = df_test[c.FEATURES]
    y_testset = df_test[c.Y_CLASSIFICATION]

    # Merkmale skalieren
    X_testset = StandardScaler().fit_transform(X_testset)

    # Dimensionsreduzierung
    # pca = PCA(n_components=n_components)
    # X_testset = pca.fit_transform(X_testset)
    # print(pca.explained_variance_ratio_)

    return X_train, X_test, y_train, y_test, X_testset, y_testset, df_test


def run_classification_models(X_train, X_test, y_train, y_test,
                              X_testset, y_testset, df_test):
    # Laufvariable für Plots
    figure = 0

    # Modelle betrachten
    for name, model in c.TUNED_CLASSIFIER.items():
        print('=' * 20)
        print('Modell:', name)

        # Modell trainieren
        clf, y_pred = cl.train_model(model,
                                     X_train, y_train, X_test)

        # Genauigkeit für Testdaten aus Trainingsdaten bestimmen
        print('Ergebnisse des Trainings:')
        cl.score_model(y_pred, y_test)

        # Feature Importance
        # Quelle: https://campus.datacamp.com/courses/
        # machine-learning-with-tree-based-models-in-python/
        # bagging-and-random-forests?ex=10
        # Create a pd.Series of features importances
        try:
            importances = pd.Series(data=clf.feature_importances_,
                                    index=c.FEATURES)

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
        # cl.tune_classifier(model=model, params=c.HYPERPARAMS,
        #                    X_train=X_train, y_train=y_train)

        # Genauigkeit für Testdatensatz bestimmen
        print('Ergebnisse des gesamten Testsets:')
        predictions = clf.predict(X_testset)
        cl.score_model(predictions, y_testset)

        # Compute predicted probabilities: y_pred_prob
        # y_pred_prob = clf.predict_proba(X_testset)[:, 1]

        # ROC-Kurven für jeweils 2 Klassen (0 & 1, 1 & 2, 0 & 2) ausgeben
        plt.figure(figure)
        # cl.show_roc_curve(y_testset, y_pred_prob,
        #                   title=f'ROC-Curve for model {name}')
        cl.show_roc_curve(y_testset, predictions,
                          title=f'ROC-Curve for model {name}')
        figure += 1

        # Betrachtung der einzelnen Bearings
        for bearing in c.TEST_SETS:
            print('Ergebnisse für {}:'.format(bearing))
            reduced_set = df_test.loc[bearing, slice(None), :]
            reduced_X = reduced_set[c.FEATURES]
            reduced_y = reduced_set[c.Y]

            # Merkmale skalieren
            reduced_X = StandardScaler().fit_transform(reduced_X)

            # Dimensionsreduzierung
            # pca = PCA(n_components=n_components)
            # reduced_X = pca.fit_transform(reduced_X)

            predictions = clf.predict(reduced_X)
            cl.score_model(predictions, reduced_y)


def prepare_regression_data():
    # Trainingsdaten zur Klassifizierung einlesen
    filepath = os.path.join(c.INPUT_CLASSIFIER, 'Lernset_Klassifikation.csv')
    df = pd.read_csv(filepath, index_col=[0, 1])
    # df.reset_index(inplace=True)

    # Labels anzeigen
    # print(df['RUL'].tail())
    # sns.scatterplot(x='Observation', y='RUL', hue='bearing', data=df)
    # plt.title('RUL visualization for trainingsset')
    # plt.show()

    # sns.scatterplot(x='TIMESTAMP', y='RUL', hue='bearing', data=df)
    # plt.title('RUL visualization for trainingsset')
    # plt.show()

    # Merkmale extrahieren
    X = df[c.FEATURES]
    y = df[c.Y_REGRESSION]

    # Merkmale skalieren
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    # Dimensionsreduzierung
    # pca = PCA(n_components=0.95)
    # X = pca.fit_transform(X)
    # global n_components
    # n_components = len(X[0])
    # print(pca.explained_variance_ratio_)

    # Train-Test-Split - ggf. stratify=y?
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        random_state=14)

    # Testdatensatz einlesen
    filepath = os.path.join(c.INPUT_CLASSIFIER, 'Testset_Klassifikation.csv')
    df_test = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X_testset = df_test[c.FEATURES]
    y_testset = df_test[c.Y_REGRESSION]

    # Merkmale skalieren
    X_testset = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_testset)

    # Dimensionsreduzierung
    # pca = PCA(n_components=n_components)
    # X_testset = pca.fit_transform(X_testset)
    # print(pca.explained_variance_ratio_)

    return X_train, X_test, y_train, y_test, X_testset, y_testset, df_test


def run_regression_models(X_train, X_test, y_train, y_test,
                          X_testset, y_testset, df_test):
    figure = 1

    # Modelle betrachten
    for name, model in c.REGRESSORS.items():
        print('=' * 20)
        print('Modell:', name)

        # Modell trainieren
        reg, y_pred = rg.train_model(model,
                                     X_train, y_train, X_test)

        # Genauigkeit für Testdaten aus Trainingsdaten bestimmen
        print('Ergebnisse des Trainings:')
        rg.score_model(y_pred, y_test)

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
            reduced_X = MinMaxScaler(feature_range=(0, 1)).fit_transform(reduced_X)

            # Dimensionsreduzierung
            # pca = PCA(n_components=n_components)
            # reduced_X = pca.fit_transform(reduced_X)

            predictions = reg.predict(reduced_X)
            rg.score_model(predictions, reduced_y)

            plt.figure(figure)
            plt.plot(reduced_y['RUL'].to_list(), color='black', label='Real')
            plt.plot(predictions, color='green', label='Predictions')
            plt.title('RUL Prediction for {} model {}'.format(bearing, name))
            plt.xlabel('Time')
            plt.ylabel('RUL')
            plt.legend()
            plt.show()
            figure += 1


def main():
    # Warnungen unterdrücken
    warnings.filterwarnings('ignore')

    # Test- und Trainingsdaten erzeugen
    # create_data()

    # Daten labeln
    # label_data()

    # Daten zur Klassifikation vorbereiten
    # X_train, X_test, y_train, y_test, X_testset, y_testset, df_test = \
    #     prepare_classification_data()

    # Klassifikation durchführen
    # run_classification_models(X_train, X_test, y_train, y_test,
    #                           X_testset, y_testset, df_test)

    # Daten zur Regression vorbereiten
    X_train, X_test, y_train, y_test, X_testset, y_testset, df_test = \
        prepare_regression_data()

    # Regression durchführen
    # run_regression_models(X_train, X_test, y_train, y_test,
    #                       X_testset, y_testset, df_test)

    # Deep-Learning
    model = c.keras_model_3()
    training_hist = model.fit(X_train, y_train, epochs=100, batch_size=8,
                              verbose=1, validation_split=0.2)
    # make predictions
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    print('Ergebnisse des Trainings:')
    # rg.score_model(trainPredict, y_train)
    rg.score_model(testPredict, y_test)

    # Genauigkeit für Testdatensatz bestimmen
    print('\nErgebnisse des gesamten Testsets:')
    predictions = model.predict(X_testset)
    # rg.score_model(predictions, y_testset)

    # Early Stopping Kurve
    plt.figure(1)
    x_cords = np.arange(100)+1
    loss = training_hist.history['loss']
    val_loss = training_hist.history['val_loss']
    plt.plot(x_cords, loss, label='Traing-Loss')
    plt.plot(x_cords, val_loss, label='Validierung-Loss')
    plt.title('Vergleich von Training und Validierungs Loss-Funktion')
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

        # Dimensionsreduzierung
        # pca = PCA(n_components=n_components)
        # reduced_X = pca.fit_transform(reduced_X)

        predictions = model.predict(reduced_X)
        rg.score_model(predictions, reduced_y)

        # print(predictions)
        # print(reduced_y)

        plt.figure(figure)
        plt.plot(reduced_y['RUL'].to_list(), color='black', label='Real')
        plt.plot(predictions, color='green', label='Predictions')
        plt.title('RUL Prediction for {}'.format(bearing))
        plt.xlabel('Time')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()
        figure += 1


if __name__ == '__main__':
    main()
