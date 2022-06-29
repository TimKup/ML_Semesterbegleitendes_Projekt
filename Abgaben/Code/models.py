# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Enthält alle normalen Modelle.
"""

# Klassifikation
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


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
                    'Gradient_Boosting': GradientBoostingClassifier(
                        learning_rate=0.1,
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

# Regressionsmodelle
REGRESSORS = {'Linear_Regression': LinearRegression(),
              'Ridge_Regression': Ridge(random_state=11),
              'Lasso_Regression': Lasso(random_state=11),
              'Elastic_Net': ElasticNet(random_state=11),
              'RandomForestRegressor': RandomForestRegressor(random_state=11),
              'XGBoost': xgb.XGBRegressor(objective='reg:squarederror',
                                          colsample_bytree=0.3,
                                          learning_rate=0.1,
                                          max_depth=5,
                                          alpha=10,
                                          n_estimators=10),
              'SVR': SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)}
