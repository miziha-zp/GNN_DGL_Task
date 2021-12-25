import os
import joblib
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score as F1
import pandas as pd
import numpy as np
import lightgbm as lgb
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from utils import cprint, evaluationAUC_F1

'''
AUC: 0.93751118
max F1: 0.6865486423126852
'''

def evaluation(y_prob, label):
    cprint('-'*100, 'green')
    print("AUC:", AUC(label, y_prob))

train_item = pd.read_csv('task1_data/train_item_feature_table.csv')
train_user = pd.read_csv('task1_data/train_user_feature_table.csv')
train = pd.read_csv('task1_data/train_e.csv')

train = train.merge(train_user, on='userid', how='left')
train = train.merge(train_item, on='itemid', how='left')


test_item = pd.read_csv('task1_data/test_item_feature_table.csv')
test_user = pd.read_csv('task1_data/test_user_feature_table.csv')
test = pd.read_csv('task1_data/test_e.csv')

test = test.merge(test_user, on='userid', how='left')
test = test.merge(test_item, on='itemid', how='left')

removelist = ['label', 'userid', 'itemid']# ,
feature_columns = [fea for fea in train.columns if fea not in removelist]
label = 'label'
categorical_feature = []

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    # 'max_depth': -1,
    'num_leaves': 32,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'verbose': 0,
    'random_state': 42,
    'n_jobs': -1,
    'lambda_l1': 1,         
    'lambda_l2': 1
}
print('train shape:', train.shape)
print('valid shape:', test.shape)

dtrain = lgb.Dataset(train[feature_columns], label=train['label'].values)
dvalid = lgb.Dataset(test[feature_columns], label=test['label'].values)


lgb_model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=50,
        categorical_feature=categorical_feature,
    )


# print(imp.tail(5))
oof_label = np.zeros(len(test))
oof_label= lgb_model.predict(test[feature_columns])
# print(oof_label)
print(oof_label.mean())
evaluationAUC_F1(oof_label, test['label'].values)
