# -*- coding: utf-8 -*-
"""
Created on Fri May 13 2022

@author: Christian T. Seidler
"""


# Pfad zum Directory des Datensatzes
DATA_DIR = r'D:\HS Albstadt\Sommersemester 2022' \
           r'\Machine Learning in Manufacturing\Projektarbeit\Datensatz' \
           r'\Test_set'

# Pfad zum Export der generiertn .csv-Files für die Zeitanalyse
OUTPUT_TIME_DOMAIN = r'D:\HS Albstadt\Sommersemester 2022' \
                     r'\Machine Learning in Manufacturing\Projektarbeit' \
                     r'\Data\Output\time-domain'

# Pfad zum Export der generiertn .csv-Files für die Frequenzanalyse
OUTPUT_FREQUENCY_DOMAIN = r'D:\HS Albstadt\Sommersemester 2022' \
                          r'\Machine Learning in Manufacturing\Projektarbeit' \
                          r'\Data\Output\frequency-domain'

# Pfad zum Export der generiertn .csv-Files für die Spektrogramme
OUTPUT_SPECTROGRAM = r'D:\HS Albstadt\Sommersemester 2022' \
                     r'\Machine Learning in Manufacturing\Projektarbeit' \
                     r'\Data\Output\spectrograms'


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
