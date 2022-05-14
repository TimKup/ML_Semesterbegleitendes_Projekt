# -*- coding: utf-8 -*-
"""
Created on Sat May 14 2022
@author: Christian T. Seidler, Timmo Kupilas
"""

import pandas as pd
import os


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