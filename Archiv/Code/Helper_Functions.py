# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:39:20 2022

@author: Christian T. Seidler

Sonstige nützliche Funktionen für den Datensatz
"""

import matplotlib.pyplot as plt
import pandas as pd
import Data_Labeling as dl


# Funktion, um grundlegende Informationen eines Dataframes auszugeben
def print_df_infos(df):
    print(df.describe(), end='\n\n')
    print(df.head(), end='\n\n')
    print(df.tail(), end='\n\n')
    print(df.info(), end='\n\n')


# Funktion, um die berechneten Signalparameter grafisch anzuzeigen
def plot_signal_params(df, figsize=(24, 25)):
    fig, axes = plt.subplots(nrows=13, ncols=1, figsize=figsize, sharex=True)

    # Bearing 1_1
    df.plot(y=[0, 1], ax=axes[0], title='Mittelwert')
    df.plot(y=[2, 3], ax=axes[1], title='Absoluter Mittelwert')
    df.plot(y=[4, 5], ax=axes[2], title='Standardabweichung')
    df.plot(y=[6, 7], ax=axes[3], title='Schiefe')
    df.plot(y=[8, 9], ax=axes[4], title='Kurtosis')
    df.plot(y=[10, 11], ax=axes[5], title='RMS')
    df.plot(y=[12, 13], ax=axes[6], title='Absoluter maximaler Wert')
    df.plot(y=[14, 15], ax=axes[7], title='Amplitude - peak-to-peak')
    df.plot(y=[16, 17], ax=axes[8], title='Crest-Faktor')
    df.plot(y=[18, 19], ax=axes[9], title='Shape-Faktor')
    df.plot(y=[20, 21], ax=axes[10], title='Impuls')
    df.plot(y=[22, 23], ax=axes[11], title='Clearance-Faktor')
    df.plot(y=[24, 25], ax=axes[12], title='Entropie')

    plt.show()


# Funktion, um die berechneten Signalparameter grafisch anzuzeigen
def plot_frequencies(df, figsize=(24, 8)):
    # Maximaler Frequenzanteil
    df.plot(x='Observation', y=['max_freq_1_x', 'max_freq_1_y'],
            title='Frequenzanteil mit der größten Magnitude',
            legend=True, figsize=figsize, subplots=True)
    plt.show()

    # Grafische Darstellung der 5 stärksten Frequenzanteile
    # in x-Richtung
    df.plot(x='Observation', y=['max_freq_1_x', 'max_freq_2_x', 'max_freq_3_x',
                                'max_freq_4_x', 'max_freq_5_x'],
            title='Frequenzanteil mit der größten Magnitude in x-Richtung',
            legend=True, figsize=figsize, subplots=True)
    plt.show()

    # in y-Richtung
    df.plot(x='Observation', y=['max_freq_1_y', 'max_freq_2_y', 'max_freq_3_y',
                                'max_freq_4_y', 'max_freq_5_y'],
            title='Frequenzanteil mit der größten Magnitude in y-Richtung',
            legend=True, figsize=figsize, subplots=True)
    plt.show()


# Funktion um die erzeugten Labels zu überprüfen
def visualize_labels(file, column=None, title=None):
    # Labels anzeigen
    df = pd.read_csv(file)

    # print(df[df.abs_rolling_mean_x >= 1.5].index.min())
    # print(df[df.abs_rolling_mean_y >= 1.5].index.min())

    df = dl.append_rul_class_col(df)
    dl.visualize_class_labeling(df)

    # rul_max = df[df['RUL_Class'] == 0]
    # print(rul_max)

    if column is None:
        df.abs_rolling_mean_x.plot(color='red')
        df.abs_rolling_mean_y.plot(color='green')
    else:
        df.column.plot(color='red')
        df.column.plot(color='green')

    if title is None:
        plt.title('Darstellung der Labels')
    else:
        plt.title(title)

    plt.show()
