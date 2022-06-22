# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler, Timmo Kupilas

Funktionen zur Labelung des Datensatzes.
"""

import pandas as pd
import os
import numpy as np


# Funktion zur automatischen Labelung der Daten in Klassen
def append_rul_class_col(df):
    # Index zurücksetzen
    df_reset = df.reset_index()

    # Dataframe durchlaufen und prüfen, wann das erste Mal 20G in x- oder
    # y-Richtung auftreten
    # Index dieser Spalte bestimmen
    try:
        # index_x = df[df.abs_max_x >= 20].index.min()
        index_x = df[df.abs_rolling_mean_x >= 2].index.min()  # 1.5
    except ValueError:
        index_x = df_reset.iloc[-1]
    try:
        # index_y = df[df.abs_max_y >= 20].index.min()
        index_y = df[df.abs_rolling_mean_y >= 2].index.min()
    except ValueError:
        index_y = df_reset.iloc[-1]

    # Bestimmen, welcher Wert kleiner ist.
    try:
        rul_0_index = np.nanmin([index_x, index_y])
        # if RuntimeWarning:
        #     rul_0_index = len(df_reset['index']) - 1
    except ValueError:
        # Für den Fall, dass der Wert nie über 20G ist, wird der letzte Wert
        # einfach auf RUL_Class=0 gesetzt
        pass
        # rul_0_index = len(df_reset['index']) - 1
    if np.isnan(rul_0_index):
        pass
        # rul_0_index = len(df_reset['index']) - 1
    # if rul_0_index == 'nan':
    #     rul_0_index = len(df_reset['index']) - 1

    # Funktion zur Berechnung der zugehörigen Klasse je nach verbleibender
    # time-to-failure
    # def calc_rul_class(row):
    #     if row['index'] >= rul_0_index:
    #         val = 0
    #     # Erster Eintrag ist immer ohne Verschleiß
    #     elif row['index'] == 0:
    #         val = 2
    #     else:
    #         diff = rul_0_index - row['index']
    #         # Zeit bis zum Ausfall in Stunden
    #         time_to_failure = (diff * 10) / 3600

    #         # Kein Verschleiß
    #         if time_to_failure > 1.225:  # 1.2
    #             val = 2
    #         else:
    #             # Verschleiß erkennbar
    #             val = 1

    #     return val

    # Variante ohne kaputt am Ende
    def calc_rul_class(row):
        # Erster Eintrag ist immer ohne Verschleiß
        if row['index'] == 0:
            val = 2
        else:
            diff = rul_0_index - row['index']
            # Zeit bis zum Ausfall in Stunden
            time_to_failure = (diff * 10) / 3600

            if np.isnan(time_to_failure):
                time_to_failure = len(df) / 360 - row['index'] / 360

            # Kein Verschleiß
            if time_to_failure > 1.225:  # 1.2
                val = 2
            elif time_to_failure <= 1.225 and time_to_failure > 0:
                # Verschleiß erkennbar
                val = 1
            else:
                # Teil kaputt
                val = 0

        return val

    def calc_rul(row):
        diff = rul_0_index - row['index']
        # Zeit bis zum Ausfall in Stunden
        time_to_failure = (diff * 10) / 3600

        if np.isnan(time_to_failure):
            time_to_failure = len(df) / 360 - row['index'] / 360

        return time_to_failure

    def calc_timestamp(row):
        # Erster Eintrag ist Nullpunkt [in Stunden]
        if row['index'] == 0:
            timestamp = 0
        else:
            timestamp = row['index'] * 10 / 3600

        return timestamp

    # RUL-Klassen anlegen
    df_reset['RUL_Class'] = df_reset.apply(calc_rul_class, axis=1)

    # RUL-Wert anlegen
    df_reset['RUL'] = df_reset.apply(calc_rul, axis=1)
    df_reset[df_reset['RUL'] < 0] = 0

    # Timestamp anfügen
    df_reset['TIMESTAMP'] = df_reset.apply(calc_timestamp, axis=1)

    return df_reset


# Funktion zum Labeln aller Datensätze und
# Erstellung eines großen Dataframes
def create_class_labeling(dataset, output_dir, testset=False):
    if testset:
        names = ['Test_set', 'Testsets']
    else:
        names = ['Learning_set', 'Lernsets']

    df_time = pd.DataFrame()
    df_freq = pd.DataFrame()
    df_result = pd.DataFrame()

    for bearing in dataset.keys():
        # Zeitdaten
        # Datenpfad anlegen
        file = os.path.join(output_dir,
                            'time-domain',
                            names[1],
                            (bearing + '_time_params.csv')
                            )

        # Daten einlesen
        data = pd.read_csv(file)

        # Labels erstellen
        data_labled = append_rul_class_col(data)

        # Multi-Index erstellen
        data_labled['bearing'] = bearing
        data_labled['index'] = data_labled.index.astype(str)
        data_labled.set_index(['bearing', 'index'], inplace=True)

        # Dataframes zusammenfügen
        df_time = pd.concat([data_labled, df_time], ignore_index=False)

        # Frequenzdaten
        # Datenpfad anlegen
        file = os.path.join(output_dir,
                            'frequency-domain',
                            names[1],
                            (bearing + '_frequency_params.csv')
                            )

        # Daten einlesen
        data = pd.read_csv(file)

        # Multi-Index erstellen
        data['bearing'] = bearing
        data['index'] = data.index.astype(str)
        data.set_index(['bearing', 'index'], inplace=True)

        # Dataframes zusammenfügen
        df_freq = pd.concat([data, df_freq], ignore_index=False)

    # Zeit- und Frequenzmerkmale zusammenfügen
    df_result = pd.concat([df_time, df_freq], axis=1)

    return df_result


# Funktion zum Anzeigen der erstellten Labels
def visualize_class_labeling(df):
    df.plot(x='index', y='RUL_Class', kind='scatter')
