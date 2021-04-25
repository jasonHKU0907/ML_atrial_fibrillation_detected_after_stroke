
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utilities import *
from xgboost import XGBClassifier
import random
import operator
import itertools
import warnings
warnings.filterwarnings(('ignore'))

dpath = '../'
X = pd.read_csv(dpath + 'Data/DatasetAF.csv', low_memory = False)
y = X['y_AF']

mykf = StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)
my_params = {'n_estimators': 400, 'max_depth': 3, 'min_child_weight': 7, 'subsample': 0.9, 'eta': 0.01}

fold_ids = pd.DataFrame(np.zeros(X.shape[0]), columns= ['fold_ids'])
i = 0
for train_idx, test_idx in mykf.split(X, y):
    fold_ids.iloc[test_idx,:] = i
    i += 1


def find_next_f(my_params, base_f, pool_f, X, y_train, y_test):
    selected_f = [pool_f[0]]
    my_f = base_f + selected_f
    my_X = X[my_f]
    my_X_train, my_X_test = my_X.iloc[train_idx,:], my_X.iloc[test_idx,:]
    my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 8, eval_metric = 'auc', scale_pos_weight = imbalance_ratio,
                           verbosity = 1, seed = 2020)
    my_xgb.set_params(**my_params)                             
    my_xgb.fit(my_X_train, y_train)
    y_pred_prob = my_xgb.predict_proba(my_X_test)[:, 1]
    auc = np.round(roc_auc_score(y_test, y_pred_prob), 4)
    update_base = my_f
    update_pool = pool_f
    update_pool.remove(pool_f[0])
    return ((auc, update_base, update_pool, selected_f))


f_df = pd.read_csv(dpath + 'Results/Sequential_Forward_Selector.csv', low_memory = False)
pool_f = f_df['features'].tolist()
base_f = []

f_id = 1
train_idx = [iter for iter in range(14146) if int(fold_ids.iloc[iter]) != f_id]
test_idx = [iter for iter in range(14146) if int(fold_ids.iloc[iter]) == f_id]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
imbalance_ratio = np.round((len(y_train) - np.sum(y_train)) / np.sum(y_train),2)

best_auc, sigma = 0, 1
my_auc, my_f = [], []
for i in range(50):
    my_params = my_params0
    update_auc, update_base, update_pool, selected_f = find_next_f(my_params, base_f, pool_f,
                                                                   X, y_train, y_test)
    my_auc.append(update_auc)
    my_f.append(selected_f)
    base_f = update_base
    pool_f = update_pool
    my_auc_df = pd.DataFrame(my_auc, columns = ['AUC'])
    my_f_df = pd.DataFrame(my_f, columns = ['features'])
    finalout = pd.concat((my_auc_df, my_f_df), axis = 1)
    finalout.to_csv(dpath + '/Results/S3_XGB_top50_fold{}.csv'.format(f_id))



my0 = pd.read_csv(dpath + 'S3_XGB_top50_fold0.csv', low_memory = False)
my1 = pd.read_csv(dpath + 'S3_XGB_top50_fold1.csv', low_memory = False)
my2 = pd.read_csv(dpath + 'S3_XGB_top50_fold2.csv', low_memory = False)
my3 = pd.read_csv(dpath + 'S3_XGB_top50_fold3.csv', low_memory = False)
my4 = pd.read_csv(dpath + 'S3_XGB_top50_fold4.csv', low_memory = False)

auc0 = np.array(my0['AUC'])
auc1 = np.array(my1['AUC'])
auc2 = np.array(my2['AUC'])
auc3 = np.array(my3['AUC'])
auc4 = np.array(my4['AUC'])

my_auc = (auc0+auc1+auc2+auc3+auc4)/5
my_idx = np.array(range(50))+1

plt.figure(figsize = (8, 8))
plt.plot(my_idx, auc0, 'green', alpha = 0.25, linewidth = 3)
plt.plot(my_idx, auc1, 'green', alpha = 0.25, linewidth = 3)
plt.plot(my_idx, auc2, 'green', alpha = 0.25, linewidth = 3)
plt.plot(my_idx, auc3, 'green', alpha = 0.25, linewidth = 3)
plt.plot(my_idx, auc4, 'green', alpha = 0.25, linewidth = 3)
plt.plot(my_idx, my_auc, 'green', alpha = 0.95, linewidth = 4)
plt.fill_between(my_idx, auc0, auc1, color = 'mediumspringgreen', alpha = 0.3)
plt.xticks(my_idx, pool_f, rotation = 90, fontsize = 35)
plt.show()

