import numpy as np
# import numpy 
import random
import scipy.special as special
import math
import pandas as pd
from math import log
import gc
import os
import joblib
from termcolor import colored

import pandas as pd
import numpy as np
from termcolor import colored
from sklearn import metrics
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score as F1

def cprint(input, color):
    print(colored(input, color))

def cprint(x, color):
    print(colored(x, color))
def get_lgbm_varimp(model, train_columns, max_vars=50):
    
    if "basic.Booster" in str(model.__class__):
        # lightgbm.basic.Booster was trained directly, so using feature_importance() function 
        cv_varimp_df = pd.DataFrame([train_columns, model.feature_importance()]).T
    else:
        # Scikit-learn API LGBMClassifier or LGBMRegressor was fitted, 
        # so using feature_importances_ property
        cv_varimp_df = pd.DataFrame([train_columns, model.feature_importances_]).T

    cv_varimp_df.columns = ['feature_name', 'varimp']

    cv_varimp_df.sort_values(by='varimp', ascending=False, inplace=True)

    cv_varimp_df = cv_varimp_df.iloc[0:max_vars]   

    return cv_varimp_df
    
    
    
def calc_precision_recall_at_k(y_true, y_pred, bounds):
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    F1s = 2 * (precisions * recalls) / (precisions + recalls + 0.0001)
    max_f1 = np.max(F1s)
    thresholds = np.array(list(thresholds) + [1])
    precision_at_recall = []
    threshold_at_recall= []
    recall_at_precision = []
    threshold_at_precision=[]
    
    
    for k in bounds:
        # precision@recall=k
        if len(precisions[recalls >= k]) > 0:
            precision_at_recall_k = precisions[recalls >= k][-1]
            threshold_at_recall_k = thresholds[recalls >= k][-1]
        else:
            precision_at_recall_k = 0
            threshold_at_recall_k = 0
        precision_at_recall.append(precision_at_recall_k)
        threshold_at_recall.append(threshold_at_recall_k)        
        
        # recall@precision=k
        if len(recalls[precisions >= k]) > 0:
            recall_at_precision_k = recalls[precisions >= k][0]
            threshold_at_precision_k = thresholds[precisions >= k][0]
        else:
            recall_at_precision_k = 0
            threshold_at_precision_k = 0
        recall_at_precision.append(recall_at_precision_k)
        threshold_at_precision.append(threshold_at_precision_k)

    roc_auc = metrics.roc_auc_score(y_true, y_pred)#=============add metric============#
    average_precision = metrics.average_precision_score(y_true, y_pred)#=============add metric============#
    # accuracy = metrics.accuracy_score(y_true, y_pred)#=============add metric============#
    pr_auc = metrics.auc(recalls, precisions)

    return precision_at_recall, threshold_at_recall, recall_at_precision, threshold_at_precision, max_f1, roc_auc, average_precision, pr_auc




def evaluationAUC_F1(y_prob, label):
    y_prob = np.array(y_prob)
    label = np.array(label)
    precisions, recalls, thresholds = metrics.precision_recall_curve(label, y_prob)
    F1s = 2 * (precisions * recalls) / (precisions + recalls + 0.0001)
    max_f1 = np.max(F1s)
    idx = 0
    for id, f1 in enumerate(F1s):
        if f1 == max_f1:
            idx = id
            break

    auc_ = AUC(label, y_prob)
    cprint('-'*100, 'green')
    print("AUC:", auc_)
    print("max F1:", max_f1)
    cprint('-'*100, 'green')

    print("best threshold:", thresholds[idx])
    return auc_, max_f1, thresholds[idx]

