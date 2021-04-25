
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utility.Processing_Utilities import *
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *
from xgboost import XGBClassifier
from itertools import product
import random
import warnings
warnings.filterwarnings(('ignore'))

dpath = '../'
X = pd.read_csv(dpath + 'Data/DatasetAF.csv', low_memory = False)
y = X['y_AF']

# Remove features from HOLTER report
# because the diagnosis golden standard of AF was obtained from HOLTER report
# Include any information from HOLTER report would conflicts the study objective

full_f = X.columns.tolist()
rm_f_keywords = ['HOLTER_']
rm_f = [f for f in full_f if any(keyword in f for keyword in rm_f_keywords)]
my_f = [f for f in full_f if f not in rm_f]
X = X[my_f]
nb_models = 1000
nb_top_models = int(nb_models*0.05)
criteria_lbd = 0.7

mykf = StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)

params_dict = {'n_estimators': np.linspace(100, 500, 9).astype('int32').tolist(),
               'max_depth': np.linspace(6, 15, 6).astype('int32').tolist(),
               'min_child_weight': np.linspace(1, 15, 6).astype('int32').tolist(),
               'subsample': np.round(np.linspace(0.6, 1.0, 5), 2).tolist(),
               'eta': [0.1, 0.05, 0.01, 0.005], 
               'colsample_bytree': np.round(np.linspace(0.8, 1.0, 5), 2).tolist()}

cutoff_list = [np.round(0.01+i*0.01, 2) for i in range(99)]
selected_params = select_params_combo(params_dict, nb_items)
myout = []

for my_params in selected_params:
    pred_probs_list, y_test_list = [], []
    for train_idx, test_idx in mykf.split(X, y):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        imbalance_ratio = np.round((len(y_train) - np.sum(y_train)) / np.sum(y_train),2)
        my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 16, eval_metric = 'auc', scale_pos_weight = imbalance_ratio,
                               verbosity = 1, seed = 2020)
        my_xgb.set_params(**my_params)
        my_xgb.fit(X_train, y_train)
        pred_probs_list.append(my_xgb.predict_proba(X_test)[:, 1])
        y_test_list.append(y_test)
    optimal_cutoff = select_cutoff(pred_probs_list, y_test_list, cutoff_list, criteria_lbd)
    eval_avg, eval_std = get_avg_score(pred_probs_list, y_test_list, optimal_cutoff)
    myout.append((optimal_cutoff, eval_avg, eval_std, my_params))
    optimal_cutoff_list = [ele[0] for ele in myout]
    optimal_cutoff_list = pd.DataFrame(optimal_cutoff_list, columns = ['cutoff'])
    results = [ele[1] for ele in myout]
    results = pd.DataFrame(results, columns = ['Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'AUC', 'APR'])
    std = [ele[2] for ele in myout]
    std = pd.DataFrame(std, columns = ['Acc_std', 'Sens_std', 'Spec_std', 'Prec_std', 'Youden_std', 'F1_std', 'AUC_std', 'APR_std'])
    params = [ele[3] for ele in myout]
    params = pd.DataFrame(params)
    models_rank = pd.concat((optimal_cutoff_list, results, std, params), axis = 1)
    #models_rank.to_csv(results_dpath + 'XGB_Models.csv')
    print('The results for now are: {}'.format(eval_avg))
    print('The selected params are: {}'.format(my_params))


def normal_imp(mydict):
'''
Normalize the importance weights for each selected models
'''
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

models_rank.sort_values(by = 'AUC', ascending = False, inplace = True)
params_dct = models_rank.iloc[:nb_top_models, 18:].to_dict()
params_lst = [dict(zip(params_dct, values)) for values in zip(*[params_dct[k].values() for k in params_dct])]


cum_tg_imp = Counter()
cum_shap_imp = np.zeros(X.shape[1])

for my_params in params_lst:
    tg_imp_cv = Counter()
    shap_imp_cv = np.zeros(X.shape[1])
    for train_idx, test_idx in mykf.split(X, y):
        X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
        imbalance_ratio = np.round((len(y_train) - np.sum(y_train)) / np.sum(y_train),2)
        my_xgb = XGBClassifier(objective = 'binary:logistic', nthread = 16, eval_metric = 'auc', scale_pos_weight = imbalance_ratio,
                               verbosity = 1, seed = 2020)
        my_xgb.set_params(**my_params)
        my_xgb.fit(X_train, y_train)
        explainer = shap.TreeExplainer(my_xgb)
        shap_values = explainer.shap_values(X_train)
        shap_values = abs(np.average(shap_values, axis = 0))
        shap_imp_cv += shap_values / np.sum(shap_values)
        totalgain_imp = my_xgb.get_booster().get_score(importance_type = 'total_gain')
        tg_imp_cv += Counter(normal_imp(totalgain_imp))
    cum_tg_imp += tg_imp_cv
    cum_shap_imp += shap_imp_cv

normal_shap_imp = list(cum_shap_imp/(5*nb_top_models))
shap_imp_df = pd.DataFrame({'Features': X.columns.tolist(),
                            'ShapValues_cv': normal_shap_imp})

normal_tg_imp = normal_imp(cum_tg_imp)
tg_imp_df = pd.DataFrame({'Features': list(normal_tg_imp.keys()), 
                          'TotalGain_cv': list(normal_tg_imp.values())})

my_imp_df = pd.merge(left = shap_imp_df, right = tg_imp_df, how = 'left')
my_imp_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)

my_imp_df.to_csv(dpath + 'Results/Features_Ranking.csv')


