# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Funktionen zum Einlesen der Daten. Zudem werden die Features in Zeit- und
Frequenzbereich aus den eingelesenen Daten extrahiert. Die gefilterten Daten
werden automatisch abgespeichert.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy import fftpack


# Funktion, um eine Messung einzulesen und als pd.Dataframe zurückzugeben
def read_single_measurement(filepath):
    """Einlesen einer einzelnen .csv-Datei."""
    # Nur die letzten beiden Spalten einlesen, da diese die Messdaten in x-
    # und y-Richtung enthalten
    measurement = pd.read_csv(filepath, sep=',', header=None, usecols=[4, 5])

    # Header umbenennen
    measurement.rename(columns={4: 'acc_x', 5: 'acc_y'}, inplace=True)

    return measurement


# Funktion, um aus den Messdaten die interessanten Informationen zu extrahieren
def calc_signal_params(data, rolling_mean_size=None):
    """Zeitsignalparameter aus Dataframe berechnen."""
    # Breite für gleitenden Mittelwert setzen
    if rolling_mean_size is None:
        rolling_mean_size = int(2560 / 4)

    cols = ['acc_x', 'acc_y']

    # Mittelwert
    mean_x, mean_y = data[cols].mean()

    # Absoluter Mittelwert
    abs_mean_x, abs_mean_y = data[cols].abs().mean()

    # Standardabweichung
    std_x, std_y = data[cols].std()

    # Schiefe - Skew
    skew_x, skew_y = data[cols].skew()

    # Kurtosis
    kurtosis_x, kurtosis_y = data[cols].kurtosis()

    # RMS
    rms_x, rms_y = np.sqrt((data[cols]**2).sum() / len(data[cols]))

    # Absoluter maximaler Wert
    abs_max_x, abs_max_y = data[cols].abs().max()

    # Peak-to-Peak
    p2p_x, p2p_y = data[cols].abs().max() - data[cols].abs().min()

    # Crest Faktor
    crest_x = abs_max_x / rms_x
    crest_y = abs_max_y / rms_y

    # Shape Faktor
    shape_x = rms_x / abs_mean_x
    shape_y = rms_y / abs_mean_y

    # Impulse
    impulse_x = abs_max_x / abs_mean_x
    impulse_y = abs_max_y / abs_mean_y

    # Clearance Faktor
    clearance_x, clearance_y = ((np.sqrt(data[cols].abs())).sum() /
                                len(data[cols]))**2

    # Entropie einfügen
    entropy_x = entropy(pd.cut(data[cols[0]], 500).value_counts())
    entropy_y = entropy(pd.cut(data[cols[1]], 500).value_counts())

    # Gleitender Mittelwert
    rolling_mean_x = np.nanmax(data[cols[0]]
                               .rolling(rolling_mean_size)
                               .mean()
                               )
    rolling_mean_y = np.nanmax(data[cols[1]]
                               .rolling(rolling_mean_size)
                               .mean()
                               )

    # Absoluter gleitender Mittelwert
    abs_rolling_mean_x = np.nanmax(data[cols[0]]
                                   .abs()
                                   .rolling(rolling_mean_size)
                                   .mean()
                                   )
    abs_rolling_mean_y = np.nanmax(data[cols[1]]
                                   .abs()
                                   .rolling(rolling_mean_size)
                                   .mean()
                                   )

    # DataFrame aufbauen
    signal_params = pd.DataFrame({'mean_acc_x': [], 'mean_acc_y': [],
                                  'abs_mean_x': [], 'abs_mean_y': [],
                                  'std_x': [], 'std_y': [],
                                  'skew_x': [], 'skew_y': [],
                                  'kurtosis_x': [], 'kurtosis_y': [],
                                  'rms_x': [], 'rms_y': [],
                                  'abs_max_x': [], 'abs_max_y': [],
                                  'p2p_x': [], 'p2p_y': [],
                                  'crest_x': [], 'crest_y': [],
                                  'shape_x': [], 'shape_y': [],
                                  'impulse_x': [], 'impulse_y': [],
                                  'clearance_x': [], 'clearance_y': [],
                                  'entropy_x': [], 'entropy_y': [],
                                  'rolling_mean_x': [], 'rolling_mean_y': [],
                                  'abs_rolling_mean_x': [],
                                  'abs_rolling_mean_y': []})

    # DataFrame mit berechneten Parametern befüllen
    signal_params.loc[0] = [mean_x, mean_y,
                            abs_mean_x, abs_mean_y,
                            std_x, std_y,
                            skew_x, skew_y,
                            kurtosis_x, kurtosis_y,
                            rms_x, rms_y,
                            abs_max_x, abs_max_y,
                            p2p_x, p2p_y,
                            crest_x, crest_y,
                            shape_x, shape_y,
                            impulse_x, impulse_y,
                            clearance_x, clearance_y,
                            entropy_x, entropy_y,
                            rolling_mean_x, rolling_mean_y,
                            abs_rolling_mean_x, abs_rolling_mean_y]

    return signal_params


# Funktion, um automatisiert alle Messungen einzulesen
def read_bearing_measurements(filepath, start=1, end=1, step=1):
    """Alle Messungen eines Bearings einlesen."""
    data = pd.DataFrame()

    # Filepath automatisiert anpassen
    for i in range(start, end+1, step):
        if i < 10:
            document = r'\acc_0000' + str(i) + '.csv'
        elif 10 <= i < 100:
            document = r'\acc_000' + str(i) + '.csv'
        elif 100 <= i < 1000:
            document = r'\acc_00' + str(i) + '.csv'
        else:
            document = r'\acc_0' + str(i) + '.csv'

        path = filepath + document

        # Messdaten einlesen
        input_df = read_single_measurement(path)

        # Extrahieren der Signalinformationen aus den Messdaten
        signal_params = calc_signal_params(input_df)

        # Anhängen der Daten an den DataFrame
        data = pd.concat([data, signal_params], ignore_index=True)

    return data


# Funktion zum Einlesen und abspeichern von Daten in der time-domain
def import_export_time_domain(dataset, output_dir, data_dir,
                              testset=False):
    """Gesamtes Trainings- bzw. Testdatenset der Zeitsignalanalyse einlesen
    und abspeichern."""
    if testset:
        names = ['Test_set', 'Testsets']
    else:
        names = ['Learning_set', 'Lernsets']

    for key in dataset.keys():
        # Datenpfad anlegen
        file = os.path.join(data_dir, names[0], key)

        # Daten einlesen
        data = read_bearing_measurements(file,
                                         start=1,
                                         end=dataset[key],
                                         step=1)

        # Output-Path anlegen
        output_path = os.path.join(output_dir, names[1],
                                   (key + '_time_params.csv'))

        # Daten als .csv-Datei abspeichern
        data.to_csv(output_path, index=False)

        # Statusbericht in Konsole schreiben
        print('Dataset {} has been saved!'.format(key))


# Funktion, um die FFT anzuwenden
def perform_fft(input_df, windowing=True):
    """FFT auf einem Dataframe anwenden."""
    f_s = 25600

    x = list(input_df['acc_x'])
    y = list(input_df['acc_y'])

    # Window anlegen
    win_x = np.kaiser(len(x), 5)
    win_y = np.kaiser(len(y), 5)

    if windowing:
        X = fftpack.fft(x * win_x)
        freqs_x = fftpack.fftfreq(len(x)) * f_s

        Y = fftpack.fft(y * win_y)
        freqs_y = fftpack.fftfreq(len(y)) * f_s
    else:
        # in x-Richtung
        X = fftpack.fft(x)
        freqs_x = fftpack.fftfreq(len(x)) * f_s

        # in y-Richtung
        Y = fftpack.fft(y)
        freqs_y = fftpack.fftfreq(len(y)) * f_s

    return X, freqs_x, Y, freqs_y


# Funktion, um automatisiert alle Messungen einzulesen und die maximale
# Frequenz zu extrahieren
def read_bearing_measurements_with_fft(filepath, start=1, end=1, step=1):
    """Alle Messungen eines Dataframes unter Verwendung der FFT einlesen."""
    data = pd.DataFrame()

    # Filepath automatisiert anpassen
    for i in range(start, end+1, step):
        if i < 10:
            document = r'\acc_0000' + str(i) + '.csv'
        elif 10 <= i < 100:
            document = r'\acc_000' + str(i) + '.csv'
        elif 100 <= i < 1000:
            document = r'\acc_00' + str(i) + '.csv'
        else:
            document = r'\acc_0' + str(i) + '.csv'

        path = filepath + document

        # Messdaten einlesen
        input_df = read_single_measurement(path)

        # Extrahieren der maximalen Frequenz aus den Messdaten mittels FFT
        X, freqs_x, Y, freqs_y = perform_fft(input_df, windowing=True)

        # Bestimmen der 5 Frequenzanteile mit der größten Magnitude
        # In x-Richtung
        # Quelle: https://stackoverflow.com/questions/6910641/
        # how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        max_freqs_index_x = np.argpartition(np.abs(X), -10)[-10:]
        max_freqs_value_x = freqs_x[max_freqs_index_x]
        max_freqs_value_x = max_freqs_value_x[np.argsort(
            freqs_x[max_freqs_index_x])]
        max_freqs_list_x = list(max_freqs_value_x)
        # Negative Frequenzen (=Spiegelfrequenzen) entfernen
        del max_freqs_list_x[0:5]

        # In y-Richtung
        max_freqs_index_y = np.argpartition(np.abs(Y), -10)[-10:]
        max_freqs_value_y = freqs_y[max_freqs_index_y]
        max_freqs_value_y = max_freqs_value_y[np.argsort(
            freqs_y[max_freqs_index_y])]
        max_freqs_list_y = list(max_freqs_value_y)
        # Negative Frequenzen (=Spiegelfrequenzen) entfernen
        del max_freqs_list_y[0:5]

        # Maximale Frequenzen in DataFrame umwandeln
        frequency_df = pd.DataFrame({'Observation': [], 'max_freq_1_x': [],
                                     'max_freq_2_x': [], 'max_freq_3_x': [],
                                     'max_freq_4_x': [], 'max_freq_5_x': [],
                                     'max_freq_1_y': [], 'max_freq_2_y': [],
                                     'max_freq_3_y': [], 'max_freq_4_y': [],
                                     'max_freq_5_y': []})
        frequency_df.loc[0] = [i, max_freqs_list_x[-1], max_freqs_list_x[-2],
                               max_freqs_list_x[-3], max_freqs_list_x[-4],
                               max_freqs_list_x[-5], max_freqs_list_y[-1],
                               max_freqs_list_y[-2], max_freqs_list_y[-3],
                               max_freqs_list_y[-4], max_freqs_list_y[-5]]

        # Anhängen der Daten an den DataFrame
        data = pd.concat([data, frequency_df], ignore_index=True)

    return data


# Funktion zum Einlesen und abspeichern von Daten in der frequency-domain
def import_export_frequency_domain(dataset, output_dir, data_dir,
                                   testset=False):
    """Gesamtes Trainings- bzw. Testdatenset der Frequenzanalyse einlesen
    und abspeichern."""
    if testset:
        names = ['Test_set', 'Testsets']
    else:
        names = ['Learning_set', 'Lernsets']

    for key in dataset.keys():
        # Datenpfad anlegen
        file = os.path.join(data_dir, names[0], key)

        # Daten einlesen
        data = read_bearing_measurements_with_fft(file,
                                                  start=1,
                                                  end=dataset[key],
                                                  step=1)

        # Output-Path anlegen
        output_path = os.path.join(output_dir, names[1],
                                   (key + '_frequency_params.csv'))

        # Daten als .csv-Datei abspeichern
        data.to_csv(output_path, index=False)

        # Statusbericht in Konsole schreiben
        print('Dataset {} has been saved!'.format(key))
