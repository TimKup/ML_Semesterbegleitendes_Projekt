# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler, Timmo Kupilas

Funktionen zur Labelung des Datensatzes.
"""

import os
import pandas as pd
import numpy as np


# Funktion zur automatischen Labelung der Daten in Klassen
def append_rul_class_col(df):
    """Ergänzen von RUL, RUL_Class und Timestamp an den Datensatz."""

    # Index zurücksetzen
    df_reset = df.reset_index()

    # Dataframe durchlaufen und prüfen, wann das erste Mal 20G in x- oder
    # y-Richtung auftreten
    # Index dieser Spalte bestimmen
    try:
        # index_x = df[df.abs_max_x >= 20].index.min()
        index_x = df[df.abs_rolling_mean_x >= 2].index.min()
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
    except ValueError:
        pass

    # Variante ohne kaputt am Ende
    def calc_rul_class(row):
        """Bestimmen der RUL-Klasse für eine Zeile des Dataframes."""

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
            if time_to_failure > 1.225:
                val = 2
            elif 1.225 >= time_to_failure > 0:
                # Verschleiß erkennbar
                val = 1
            else:
                # Teil kaputt
                val = 0

        return val

    def calc_rul(row):
        """Bestimmen der RUL in Stunden für eine Zeile des Dataframes."""

        diff = rul_0_index - row['index']
        # Zeit bis zum Ausfall in Stunden
        time_to_failure = (diff * 10) / 3600

        if np.isnan(time_to_failure):
            time_to_failure = len(df) / 360 - row['index'] / 360

        return time_to_failure

    def calc_timestamp(row):
        """Bestimmen der verstrichenen Zeit in Stunden seit Messbeginn."""

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
    """Gesamten Trainings- bzw. Testdatensatz labeln."""

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
    """Visualisierung der Spalte 'RUL_Class'."""

    df.plot(x='index', y='RUL_Class', kind='scatter')
