
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
import operator
import itertools
import warnings
warnings.filterwarnings(('ignore'))

dpath = '/../'

X = pd.read_csv(dpath + 'Data/DatasetAF.csv', low_memory = False)
y = X['y_AF']
f_df = pd.read_csv(dpath+ 'Results/Features_Ranking.csv', low_memory = False)
f_df = f_df.sort_values(by = 'TotalGain_cv', ascending = False)
base_f = []
pool_f = list(f_df['Features'])[:nb_top_features]
nb_top_features = 50
my_params0 = {'n_estimators': 400, 'max_depth': 3, 'min_child_weight': 7, 'subsample': 0.9, 'eta': 0.01}
mykf = StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)

def find_next_f(my_params, mykf, base_f, pool_f, X, y):
    AUC_list = []
    for f in pool_f:
        my_f = base_f + [f]
        my_X = X[my_f]
        AUC_cv = []
        for train_idx, test_idx in mykf.split(my_X, y):
            X_train, X_test = my_X.iloc[train_idx,:], my_X.iloc[test_idx,:]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            imbalance_ratio = np.round((len(y_train) - np.sum(y_train)) / np.sum(y_train),2)
            my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 16, eval_metric = 'auc', scale_pos_weight = imbalance_ratio,
                                   verbosity = 1, seed = 2020)
            my_xgb.set_params(**my_params)
            my_xgb.fit(X_train, y_train)
            y_pred_prob = my_xgb.predict_proba(X_test)[:, 1]
            AUC_cv.append(roc_auc_score(y_test, y_pred_prob))
        AUC_list.append(np.round(np.average(AUC_cv), 4))
    index, best_auc = max(enumerate(AUC_list), key = operator.itemgetter(1))
    selected_f = pool_f[index]
    update_base = base_f + [selected_f]
    update_pool = pool_f
    update_pool.remove(selected_f)
    return ((best_auc, update_base, update_pool, selected_f))

best_auc, sigma = 0, 1
my_auc, my_f = [], []
for i in range(nb_top_features):
    my_params = my_params0
    update_auc, update_base, update_pool, selected_f = find_next_f(my_params, mykf, base_f, pool_f, X, y)
    my_auc.append(update_auc)
    my_f.append(selected_f)
    base_f = update_base
    pool_f = update_pool
    my_auc_df = pd.DataFrame(my_auc, columns = ['AUC'])
    my_f_df = pd.DataFrame(my_f, columns = ['features'])
    finalout = pd.concat((my_auc_df, my_f_df), axis = 1)
    finalout.to_csv(dpath + 'Results/Sequential_Forward_Selector.csv')


