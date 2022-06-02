# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Enthält alle Einstellungsparameter für die Berechnung des Modells.
"""

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier


# Pfad zum Directory des Datensatzes
DATA_DIR = r'D:\HS Albstadt\Sommersemester 2022' \
           r'\Machine Learning in Manufacturing' \
           r'\Semesterbegleitendes Projekt\Data\Input'

# Pfad zum Export der generiertn .csv-Files für die Zeitanalyse
OUTPUT_TIME_DOMAIN = r'D:\HS Albstadt\Sommersemester 2022' \
                     r'\Machine Learning in Manufacturing' \
                     r'\Semesterbegleitendes Projekt\Data' \
                     r'\Output\time-domain'

# Pfad zum Export der generiertn .csv-Files für die Frequenzanalyse
OUTPUT_FREQUENCY_DOMAIN = r'D:\HS Albstadt\Sommersemester 2022' \
                          r'\Machine Learning in Manufacturing' \
                          r'\Semesterbegleitendes Projekt\Data' \
                          r'\Output\frequency-domain'

# Pfad, welche den Input für die Labelung enthält
INPUT_LABELING = OUTPUT_TIME_DOMAIN

# Pfad, welche den Order des Outputs enthält
OUTPUT_DIR = r'D:\HS Albstadt\Sommersemester 2022' \
             r'\Machine Learning in Manufacturing' \
             r'\Semesterbegleitendes Projekt\Data\Output'

# Pfad, welcher den Output der Labels darstellt
OUTPUT_LABELING = r'D:\HS Albstadt\Sommersemester 2022' \
                  r'\Machine Learning in Manufacturing' \
                  r'\Semesterbegleitendes Projekt\Data\Output\Classification'

# Pfad zum Export der generiertn .csv-Files für die Spektrogramme
# OUTPUT_SPECTROGRAM = r'D:\HS Albstadt\Sommersemester 2022' \
#                      r'\Machine Learning in Manufacturing\Projektarbeit' \
#                      r'\Data\Output\spectrograms'

# Datensets für Klassifizierung
INPUT_CLASSIFIER = r'D:\HS Albstadt\Sommersemester 2022' \
                   r'\Machine Learning in Manufacturing' \
                   r'\Semesterbegleitendes Projekt\Data\Output\Classification'

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

# Features
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
            'max_freq_5_y']

# Zielgröße
Y = ['RUL_Class']

# Klassifikationsmodelle
CLASSIFIER = {'Linear_SVC': SVC(kernel="linear", C=0.025, random_state=14),
              'Nonlinear_SVC': SVC(kernel="rbf", gamma=2, C=1,
                                   random_state=14),
              'Random_Forest':
              RandomForestClassifier(max_depth=10,
                                     n_estimators=20,
                                     max_features=1,
                                     random_state=14
                                     ),
              'AdaBoost': AdaBoostClassifier(random_state=14),
              'KNeighbors': KNeighborsClassifier(100),
              'GaussianProcess': GaussianProcessClassifier(1.0 * RBF(1.0)),
              'Decision_Tree': DecisionTreeClassifier(random_state=14,
                                                      criterion='gini',
                                                      max_depth=4,
                                                      min_samples_leaf=0.1,
                                                      min_samples_split=0.1),
              'MLPC': MLPClassifier(alpha=1, max_iter=1000, random_state=14),
              'GaussianNB': GaussianNB(),
              'QuadraticDiscriminantAnalysis':
                  QuadraticDiscriminantAnalysis(),
              'XGBoost': GradientBoostingClassifier(random_state=14)
              }

# Aktuell verwendete Modelle
ACTIVE_CLASSIFIER = {'Bagging_Classifier':
                     BaggingClassifier(base_estimator=
                                       CLASSIFIER['Decision_Tree'],
                                       n_estimators=100,
                                       random_state=14)
                     }

# Parameter für Hyperparameter-Tuning
HYPERPARAMS = {'max_depth': list(range(2, 21, 1)),
               'criterion': ['gini', 'entropy'],
               'min_samples_split': [0.1, 0.25, 0.5, 0.75, 1.0],
               'min_samples_leaf': [0.1, 0.25, 0.45]}

# Modelle mit getunten Parametern
TUNED_CLASSIFIER = {'Random_Forest':
                    RandomForestClassifier(max_depth=15,
                                           n_estimators=25,
                                           max_features=4,
                                           random_state=14
                                           ),
                    'MLPC':
                    MLPClassifier(alpha=0.001,
                                  max_iter=900,
                                  activation='logistic',
                                  random_state=14),
                    'AdaBoost':
                    AdaBoostClassifier(n_estimators=100,
                                       learning_rate=0.1,
                                       algorithm='SAMME',
                                       random_state=14),
                    'XGBoost': GradientBoostingClassifier(learning_rate=0.1,
                                                          max_depth=5,
                                                          max_features='sqrt',
                                                          n_estimators=300,
                                                          random_state=12),
                    'Decision_Tree':
                        DecisionTreeClassifier(random_state=14,
                                               criterion='gini',
                                               max_depth=4,
                                               min_samples_leaf=0.1,
                                               min_samples_split=0.1)
                    }
