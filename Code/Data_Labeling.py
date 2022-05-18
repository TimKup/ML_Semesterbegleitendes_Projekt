# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler, Timmo Kupilas

Funktionen zur Labelung des Datensatzes.
"""

import pandas as pd
import os


# Funktion zur automatischen Labelung der Daten in Klassen
def append_rul_class_col(df):
    # Index zurücksetzen
    df_reset = df.reset_index()

    # Dataframe durchlaufen und prüfen, wann das erste Mal 20G in x- oder
    # y-Richtung auftreten
    # Index dieser Spalte bestimmen
    try:
        index_x = min(df[df.abs_max_x > 20].index)
    except ValueError:
        index_x = df_reset.iloc[-1]
    try:
        index_y = min(df[df.abs_max_y > 20].index)
    except ValueError:
        index_y = df_reset.iloc[-1]

    # Bestimmen, welcher Wert kleiner ist.
    try:
        rul_0_index = min(index_x, index_y)
    except ValueError:
        # Für den Fall, dass der Wert nie über 20G ist, wird der letzte Wert
        # einfach auf RUL_Class=0 gesetzt
        rul_0_index = len(df_reset['index']) - 1

    # Funktion zur Berechnung der zugehörigen Klasse je nach verbleibender
    # time-to-failure
    def calc_rul_class(row):
        if row['index'] >= rul_0_index:
            val = 0
        else:
            diff = rul_0_index - row['index']
            # Zeit bis zum Ausfall in Stunden
            time_to_failure = (diff * 10) / 3600

            # Kein Verschleiß
            if time_to_failure > 1.2:
                val = 2
            else:
                # Verschleiß erkennbar
                val = 1

        return val

    # RUL-Klassen anlegen
    df_reset['RUL_Class'] = df_reset.apply(calc_rul_class, axis=1)

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
