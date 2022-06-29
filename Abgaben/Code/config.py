# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Enthält alle Einstellungsparameter für die Berechnung des Modells.
"""

# Pfad zum Directory des Datensatzes
DATA_DIR = r'D:\HS Albstadt\Sommersemester 2022' \
           r'\Machine Learning in Manufacturing' \
           r'\Semesterbegleitendes Projekt\Data\Input'

# Pfad zum Export der generiertn .csv-Files für die Zeitanalyse
OUTPUT_TIME_DOMAIN = r'D:\HS Albstadt\Sommersemester 2022' \
                     r'\Machine Learning in Manufacturing\Finale Version' \
                     r'\ML in Manufacturing - final\Data\Input\time-domain'

# Pfad zum Export der generiertn .csv-Files für die Frequenzanalyse
OUTPUT_FREQUENCY_DOMAIN = r'D:\HS Albstadt\Sommersemester 2022' \
                          r'\Machine Learning in Manufacturing' \
                          r'\Finale Version\ML in Manufacturing - final' \
                          r'\Data\Input\frequency-domain'

# Pfad, welche den Input für die Labelung enthält
INPUT_LABELING = OUTPUT_TIME_DOMAIN

# Pfad, welche den Order des Outputs enthält
OUTPUT_DIR = r'D:\HS Albstadt\Sommersemester 2022' \
             r'\Machine Learning in Manufacturing' \
             r'\Semesterbegleitendes Projekt\Data\Output'

# Pfad, welcher den Output der Labels darstellt
OUTPUT_LABELING = r'D:\HS Albstadt\Sommersemester 2022' \
                  r'\Machine Learning in Manufacturing\Finale Version' \
                  r'\ML in Manufacturing - final\Data\Input'

# Datensets für Modelle
INPUT_MODELS = OUTPUT_LABELING

# Länge und Name der einzelnen Datensets
LEARNING_SETS = {'Bearing1_1': 2803,
                 'Bearing1_2': 871,
                 'Bearing2_1': 911,
                 'Bearing2_2': 797,
                 'Bearing3_1': 515,
                 'Bearing3_2': 1637}

TEST_SETS = {'Bearing1_3': 1802,
             'Bearing1_4': 1139,
             'Bearing1_5': 2302,
             'Bearing1_6': 2302,
             'Bearing1_7': 1502,
             'Bearing2_3': 1202,
             'Bearing2_4': 612,
             'Bearing2_5': 2002,
             'Bearing2_6': 572,
             'Bearing2_7': 172,
             'Bearing3_3': 352}

# Features - TIMESTAMP für Klassifikation entfernen
FEATURES = ['mean_acc_x', 'mean_acc_y',
            'abs_mean_x', 'abs_mean_y',
            'std_x', 'std_y',
            'skew_x', 'skew_y',
            'kurtosis_x', 'kurtosis_y',
            'rms_x', 'rms_y',
            'abs_max_x', 'abs_max_y',
            'p2p_x', 'p2p_y',
            'crest_x', 'crest_y',
            'shape_x', 'shape_y',
            'impulse_x', 'impulse_y',
            'clearance_x', 'clearance_y',
            'entropy_x', 'entropy_y',
            'max_freq_1_x',
            'max_freq_2_x',
            'max_freq_3_x',
            'max_freq_4_x',
            'max_freq_5_x',
            'max_freq_1_y',
            'max_freq_2_y',
            'max_freq_3_y',
            'max_freq_4_y',
            'max_freq_5_y',
            'TIMESTAMP']
FEATURES_CLASSIFICATION = FEATURES[:-1]

# Zielgröße
Y_CLASSIFICATION = ['RUL_Class']
Y_REGRESSION = ['RUL']
