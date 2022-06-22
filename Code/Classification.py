# -*- coding: utf-8 -*-
"""
Created on Tue May 17 2022

@author: Christian T. Seidler

Funktionen für die Klassifikationsaufgabe
"""

# Bibliotheken importieren
from sklearn import metrics
import numpy as np

from datetime import datetime
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from itertools import cycle


# Funktion für Erstellung des Modells
def train_model(clf, X_train, y_train, X_test):

    # Modell mit Daten trainieren
    clf.fit(X_train, np.ravel(y_train))

    # Testdaten vorhersagen
    y_pred = clf.predict(X_test)

    return clf, y_pred


# Funktion zum Testen des Modells
def score_model(y_pred, y_test):
    recall = metrics.recall_score(y_test, y_pred, average='weighted',
                                  zero_division=1)
    precision = metrics.precision_score(y_test, y_pred, average='weighted',
                                        zero_division=1)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average='weighted',
                                zero_division=1)

    # Standardmetriken
    print('Recall:', recall)
    print('Precision:', precision)
    print('Accuracy:', accuracy)
    print('F1-Score:', f1_score)

    # Konfusionsmatrix
    print(metrics.confusion_matrix(y_test, y_pred))

    # Zusammenfassende Statistik
    print(metrics.classification_report(y_test, y_pred, zero_division=1))


# Funktion zum Tunen der Parameter eines Klassifikationsmodells
def tune_classifier(model, params, X_train, y_train, cv=None):
    # Quelle: https://machinelearningmastery.com/hyperparameter-optimization-
    # with-random-search-and-grid-search/
    print("Start of Tuning:", datetime.now())
    # define evaluation
    if cv is None:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    # define search
    search = GridSearchCV(model,
                          params,
                          scoring='accuracy',
                          n_jobs=-1,
                          cv=cv,
                          verbose=3)
    # execute search
    result = search.fit(X_train, np.ravel(y_train))
    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    print("End of Tuning:", datetime.now())


# Funktion zur Erstellung der ROC-Kurve
def show_roc_curve(y_testset, predictions, title=None):
    # Multiindex zurücksetzen
    roc_set = y_testset.reset_index()

    # Index für Klassen extrahieren
    index_0 = roc_set[roc_set.RUL_Class == 0].index
    index_1 = roc_set[roc_set.RUL_Class == 1].index
    index_2 = roc_set[roc_set.RUL_Class == 2].index

    # Wahre y-Werte bestimmen
    y_true_0 = roc_set.iloc[index_0].RUL_Class.to_numpy()
    y_true_1 = roc_set.iloc[index_1].RUL_Class.to_numpy()
    y_true_2 = roc_set.iloc[index_2].RUL_Class.to_numpy()

    # Vorhergesagte y-Werte bestimmen
    y_score_0 = predictions[index_0]
    y_score_1 = predictions[index_1]
    y_score_2 = predictions[index_2]

    # ROC-Kurve erzeugen
    y_score = label_binarize(y_testset, classes=[0, 1, 2])
    n_classes = y_score.shape[1]
    y_test = label_binarize(predictions, classes=[0, 1, 2])

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plt.figure()
    lw = 2
    # colors = ['darkorange', 'red', 'violet']
    # for i in range(n_classes):
    #     plt.plot(
    #         fpr[i],
    #         tpr[i],
    #         color=colors[i],
    #         lw=lw,
    #         label=f"ROC curve for class label {i} (area = %0.2f)" % roc_auc[i]
    #     )
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # # plt.show()

    # ns_fpr, ns_tpr, _ = roc_curve(np.concatenate([y_true_0, y_true_1]),
    #                               np.concatenate([y_score_0, y_score_1]))
    # print("AUC Class 0 & 1: {}".format(roc_auc_score(
    #     np.concatenate([y_true_0, y_true_1]),
    #     np.concatenate([y_score_0, y_score_1]))))
    # # plot the roc curve for the model
    # plt.plot(ns_fpr, ns_tpr, color='orange', label='Class 0 & 1', marker='.')

    # ns_fpr, ns_tpr, _ = roc_curve(np.concatenate([y_true_1, y_true_2]),
    #                               np.concatenate([y_score_1, y_score_2]),
    #                               pos_label=2)
    # print("AUC Class 1 & 2: {}".format(roc_auc_score(
    #     np.concatenate([y_true_1, y_true_2]),
    #     np.concatenate([y_score_1, y_score_2]))))
    # plt.plot(ns_fpr, ns_tpr, color='red', label='Class 1 & 2', marker='.')

    # ns_fpr, ns_tpr, _ = roc_curve(np.concatenate([y_true_0, y_true_2]),
    #                               np.concatenate([y_score_0, y_score_2]),
    #                               pos_label=2)
    # print("AUC Class 0 & 2: {}".format(roc_auc_score(
    #     np.concatenate([y_true_0, y_true_2]),
    #     np.concatenate([y_score_0, y_score_2]))))
    # plt.plot(ns_fpr, ns_tpr, color='violet', label='Class 0 & 2', marker='.')

    # Reference Line
    # plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    # plt.plot()

    # axis labels
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')

    # # Titel setzen
    # if title is None:
    #     plt.title('ROC-Curve for Testset')
    # else:
    #     plt.title(title)

    # Show Legend
    # plt.legend()

    # Show the plot
    # plt.show()

    # Quelle: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title is None:
        plt.title('ROC-Curve for Testset')
    else:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
