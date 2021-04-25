
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utility.Processing_Utilities import *
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import random
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
imp_method = 'MEAN'
mykf = StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)

params_dict = {'n_estimators': np.linspace(100, 500, 9).astype('int32').tolist(),
               'criterion': ['gini', 'entropy'],
               'max_depth': np.linspace(3, 15, 6).astype('int32').tolist(),
               'min_samples_leaf': [3, 5, 7, 10, 15, 20, 30],
               'min_samples_split': [3, 5, 7, 10, 15, 20, 30],
               'max_features': ['auto', 'sqrt', 'log2']}
               
selected_params = select_params_combo(params_dict, nb_models)


def Randomized_cv_rf(selected_params, imp_method, mykf, X, y):
    AUC_cv = []
    for train_idx, test_idx in mykf.split(X, y):
        X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        under = RandomUnderSampler(sampling_strategy = 1, random_state = 2020)
        X_train, y_train = under.fit_resample(X_train, y_train)
        imputer = my_imputer(method = imp_method)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        AUC_list = []
        for my_params in selected_params:
            my_clf = RandomForestClassifier(random_state = 2020)
            my_clf.set_params(**my_params)
            my_clf.fit(X_train, y_train)
            y_pred_prob = my_clf.predict_proba(X_test)[:, 1]
            AUC_list.append(roc_auc_score(y_test, y_pred_prob))
        AUC_cv.append(AUC_list)
    AUC_cv = np.round(np.average(np.array(AUC_cv), axis = 0),4).tolist()
    index, best_auc = max(enumerate(AUC_cv), key = operator.itemgetter(1))
    best_params = selected_params[index]
    return((best_auc, best_params))



best_auc, best_params = Randomized_cv_rf(selected_params, imp_method, mykf, X[my_f], y)


cutoff_list = [np.round(0.01+i*0.01, 2) for i in range(99)]
results_cv, output_lst = [], []

for train_idx, test_idx in mykf.split(X, y):
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx,:]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    under = RandomUnderSampler(sampling_strategy = 1, random_state = 2020)
    X_train, y_train = under.fit_resample(X_train, y_train)
    imputer = my_imputer(method = imp_method)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    my_clf = RandomForestClassifier(random_state = 2020)
    my_clf.set_params(**best_params)
    my_clf.fit(X_train, y_train)
    y_pred_prob = my_clf.predict_proba(X_test)[:, 1]
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    output_lst.append(y_pred_prob)
    output_lst.append(y_test)


params_df = pd.DataFrame.from_dict(best_params, orient = 'index')
params_df.to_csv(dpath+ 'Results/S3_RF_params.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath+ 'Results/S3_RF.csv')

output_df = pd.DataFrame(output_lst)
output_df.to_csv(dpath+ 'Results/S3_RF_Pred_Probs.csv')




