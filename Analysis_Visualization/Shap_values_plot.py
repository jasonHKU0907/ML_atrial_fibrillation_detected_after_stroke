
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
#from Utilities import *
from xgboost import XGBClassifier
from itertools import product
import random
import warnings
import shap
import matplotlib.pyplot as plt
warnings.filterwarnings(('ignore'))


dpath = '/public/home/ncrc50002/data/fudan/AF_Project/'
results_dpath = dpath + 'Models/XGB/Results/'
X = pd.read_csv(dpath + 'Data/DatasetAF.csv', low_memory = False)
y = X['y_AF']
my_f = ['ECG_PR', 'AGE', 'UCC_TTE_LAD', 'ECG_HR', 'IMG_IS_CII']
X = X[my_f]
mykf = StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

my_params = {'n_estimators': 400, 'max_depth': 3, 'min_child_weight': 7, 'subsample': 0.9, 'eta': 0.01}

for train_idx, test_idx in mykf.split(X, y):
    X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]
    imbalance_ratio = np.round((len(y_train) - np.sum(y_train)) / np.sum(y_train),2)

my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 8, eval_metric = 'auc', scale_pos_weight = imbalance_ratio,
                       verbosity = 1, seed = 2020)
my_xgb.set_params(**my_params)
my_xgb.fit(X_train, y_train)
explainer = shap.TreeExplainer(my_xgb)

shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type = 'bar')
shap_interaction_values = explainer.shap_interaction_values(X_test)
shap.summary_plot(shap_interaction_values, X_test)
shap_values = explainer.shap_values(X_test)

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_test['ECG_PR'], shap_values[:, 0], marker = 's')
plt.show()


fig = plt.figure(figsize = (8, 5))
plt.scatter(X_train['AGE'], shap_values[:, 1], marker = 's')
plt.show()


fig = plt.figure(figsize = (8, 5))
plt.scatter(X_train['UCC_TTE_LAD'], shap_values[:, 2], marker = 's')
plt.show()


fig = plt.figure(figsize = (8, 5))
plt.scatter(X_train['ECG_HR'], shap_values[:, 3], marker = 's')
plt.show()

fig = plt.figure(figsize = (8, 5))
plt.scatter(X_train['IMG_IS_CII'], shap_values[:, 4], marker = 's')
plt.show()

