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

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():

    # Trainingsdaten im Zeitbereich erstellen
    # fe.import_export_time_domain(c.LEARNING_SETS, c.OUTPUT_TIME_DOMAIN,
    #                              c.DATA_DIR, testset=False)

    # Testdaten im Zeitbereich erstellen
    # fe.import_export_time_domain(c.TEST_SETS, c.OUTPUT_TIME_DOMAIN,
    #                              c.DATA_DIR, testset=True)

    # Trainingsdaten im Frequenzbereich erstellen
    # fe.import_export_frequency_domain(c.LEARNING_SETS,
    #                                   c.OUTPUT_FREQUENCY_DOMAIN,
    #                                   c.DATA_DIR,  testset=False)

    # Testdaten im Frequenzbereich erstellen
    # fe.import_export_frequency_domain(c.TEST_SETS,
    #                                   c.OUTPUT_FREQUENCY_DOMAIN,
    #                                   c.DATA_DIR,  testset=True)

    # Trainingsdaten labeln und abspeichern
    # train_data = dl.create_class_labeling(c.LEARNING_SETS,
    #                                       c.OUTPUT_DIR,
    #                                       testset=False)
    # save_path = os.path.join(c.OUTPUT_LABELING, 'Lernset_Klassifikation.csv')
    # train_data.to_csv(save_path)

    # Testdaten labeln und abspeichern
    # test_data = dl.create_class_labeling(c.TEST_SETS,
    #                                      c.OUTPUT_DIR,
    #                                      testset=True)
    # save_path = os.path.join(c.OUTPUT_LABELING, 'Testset_Klassifikation.csv')
    # test_data.to_csv(save_path)

    # # Labels anzeigen
    # file = os.path.join(c.INPUT_LABELING, 'Lernsets',
    #                     'Bearing1_1_time_params.csv')
    # df = pd.read_csv(file)
    # df = dl.append_rul_class_col(df)
    # # dl.visualize_class_labeling(df)

    # rul_max = df[df['RUL_Class'] == 0]
    # print(rul_max)

    # raise KeyboardInterrupt()

    # Trainingsdaten zur Klassifizierung einlesen
    filepath = os.path.join(c.INPUT_CLASSIFIER, 'Lernset_Klassifikation.csv')
    df = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X = df[c.FEATURES]
    y = df[c.Y]

    # Merkmale skalieren
    X = StandardScaler().fit_transform(X)

    # TODO: Dimensionsreduzierung?

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=14)

    # Testdatensatz einlesen
    filepath = os.path.join(c.INPUT_CLASSIFIER, 'Testset_Klassifikation.csv')
    df_test = pd.read_csv(filepath, index_col=[0, 1])

    # Merkmale extrahieren
    X_testset = df_test[c.FEATURES]
    y_testset = df_test[c.Y]

    # Merkmale skalieren
    X_testset = StandardScaler().fit_transform(X_testset)

    # Modelle betrachten
    for name, model in c.CLASSIFIER.items():
        print('=' * 20)
        print('Modell:', name)

        # Modell trainieren
        clf, y_pred = cl.train_model(model,
                                     X_train, y_train, X_test)

        # Genauigkeit für Testdaten aus Trainingsdaten bestimmen
        print('Ergebnisse des Trainings:')
        cl.score_model(y_pred, y_test)

        # TODO: Hyperparamter-Tuning implementieren

        # Genauigkeit für Testdatensatz bestimmen
        print('Ergebnisse des Testsets:')
        predictions = clf.predict(X_testset)
        cl.score_model(predictions, y_testset)

        # TODO: ROC-Kurven für jeweils 2 Klassen (0 & 1, 1 & 2)

        # TODO: Testset für die Bearings einzeln betrachten


if __name__ == '__main__':
    main()
