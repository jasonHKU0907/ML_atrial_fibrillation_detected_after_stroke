
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utility.Processing_Utilities import *
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *
from xgboost import XGBClassifier
import random
import itertools
import operator
import warnings
warnings.filterwarnings(('ignore'))

dpath = '../'
X = pd.read_csv(dpath + 'Data/DatasetAF.csv', low_memory = False)
y = X['y_AF']
f_df = pd.read_csv(dpath + 'Results/Sequential_Forward_Selector.csv')
my_f = f_df['features'].tolist()[:5]
X = X[my_f]

nb_models = 100
mykf = StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)


params_dict = {'n_estimators': np.linspace(100, 500, 9).astype('int32').tolist(),
               'max_depth': np.linspace(6, 15, 6).astype('int32').tolist(),
               'min_child_weight': np.linspace(1, 15, 6).astype('int32').tolist(),
               'subsample': np.round(np.linspace(0.6, 1.0, 5), 2).tolist(),
               'eta': [0.1, 0.05, 0.01, 0.005]}

cutoff_list = [np.round(0.1+i*0.01, 2) for i in range(71)]
selected_params = select_params_combo(params_dict, nb_models)
AUC_list = []

for my_params in selected_params:
    AUC_cv = []
    for train_idx, test_idx in mykf.split(X, y):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        imbalance_ratio = np.round((len(y_train) - np.sum(y_train)) / np.sum(y_train),2)
        my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 8, eval_metric = 'auc', scale_pos_weight = imbalance_ratio,
                               verbosity = 1, seed = 2020)
        my_xgb.set_params(**my_params)
        my_xgb.fit(X_train, y_train)
        y_pred_prob = my_xgb.predict_proba(X_test)[:, 1]
        AUC_cv.append(roc_auc_score(y_test, y_pred_prob))
    AUC_list.append(np.round(np.average(AUC_cv), 4))

index, best_auc = max(enumerate(AUC_list), key = operator.itemgetter(1))
best_params = selected_params[index]

cutoff_list = [np.round(0.01+i*0.01, 2) for i in range(99)]
results_cv, output_lst = [], []

for train_idx, test_idx in mykf.split(X, y):
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    imbalance_ratio = np.round((len(y_train) - np.sum(y_train)) / np.sum(y_train),2)
    my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 8, eval_metric = 'auc', scale_pos_weight = imbalance_ratio,
                           verbosity = 1, seed = 2020)
    my_xgb.set_params(**best_params)
    my_xgb.fit(X_train, y_train)
    y_pred_prob = my_xgb.predict_proba(X_test)[:, 1]
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    output_lst.append(y_pred_prob)
    output_lst.append(y_test)

params_df = pd.DataFrame.from_dict(best_params, orient = 'index')
params_df.to_csv(dpath+ 'Results/S3_XGB_params.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath+ 'Results/S3_XGB.csv')

output_df = pd.DataFrame(output_lst)
output_df.to_csv(dpath+ 'Results/S3_XGB_Pred_Probs.csv')

