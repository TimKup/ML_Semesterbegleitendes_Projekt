# -*- coding: utf-8 -*-
"""
Created on Sat May 14 2022

@author: Christian T. Seidler, Timmo Kupilas
"""

import pandas as pd
import os

# Konfigurationsdateien einlesen
import Config as c


# Erstes Bearing einlesen
# Input Pfad
input_path = c.OUTPUT_TIME_DOMAIN

# .csv-Datei des ersten Bearings einlesen
filepath = os.path.join(input_path, 'Lernsets', 'Bearing1_1_time_params.csv')
df = pd.read_csv(filepath, sep=',', usecols=['abs_max_x', 'abs_max_y'])

# print(df.head())

# Dataframe durchlaufen und prüfen, wann das erste Mal 20G in x- oder
# y-Richtung auftreten
# Index dieser Spalte bestimmen
# index_x = min(df[df.abs_max_x > 20].index)
# index_y = min(df[df.abs_max_y > 20].index)

# # Bestimmen, welcher Wert kleiner ist
# rul_0_index = min(index_x, index_y)

# # Index zurücksetzen
# df_reset = df.reset_index()

# Neue Spalte erzeugen
# df_reset['RUL'] = df_reset.apply(calc_rul_class, axis=1)

# print(df_reset.head(10))
# print(df_reset.tail(10))


# df_reset.plot(x='index', y='RUL', kind='scatter')

# Funktion zur automatischen Labelung der Daten
def append_rul_col(df):
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
        # einfach auf RUL=0 gesetzt
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
            if time_to_failure > 2:
                val = 2
            else:
                # Verschleiß erkennbar
                val = 1

        return val

    # RUL-Klassen anlegen
    df_reset['RUL'] = df_reset.apply(calc_rul_class, axis=1)

    return df_reset


# Testlauf
df_test = append_rul_col(df)
print(df_test.head())
print(df_test.tail())
df_test.plot(x='index', y='RUL', kind='scatter')


# Bearing 1_5
filepath = os.path.join(input_path, 'Testsets', 'Bearing1_5_time_params.csv')
df = pd.read_csv(filepath, sep=',', usecols=['abs_max_x', 'abs_max_y'])
df_test = append_rul_col(df)
df_test.plot(x='index', y='RUL', kind='scatter')
