
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utility.Processing_Utilities import *
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.models import Model, Sequential
from keras.layers import *
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

mykf = StratifiedKFold(n_splits = 5, random_state = 2020, shuffle = True)
imp_method = 'MEAN'

def my_imputer(method = 'MEAN'):
    if method == 'MEAN':
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    elif method == 'MEDIAN':
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
    elif method == 'KNN':
        imputer = KNNImputer(n_neighbors = 10, weights = 'uniform')
    elif method == 'MICE':
        imputer = IterativeImputer(max_iter = 10, random_state = 2020)
    return imputer

def ann_model():
    model = Sequential()
    model.add(Flatten(input_shape = (5, 1)))
    model.add(BatchNormalization())
    model.add(Dense(4,init = 'uniform',  activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(3,init = 'uniform',  activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(1,init = 'uniform',  activation = 'sigmoid'))
    return model

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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.expand_dims(X_train, axis = -1)
    X_test = np.expand_dims(X_test, axis = -1)
    K.clear_session()
    model = ann_model()
    model.compile(loss = 'binary_crossentropy', optimizer = SGD(1e-1))
    model.fit(X_train, y_train, epochs = 80, batch_size = 32, verbose = 1)
    y_pred_prob = model.predict(X_test)
    y_pred_prob = np.reshape(y_pred_prob, (y_pred_prob.shape[0],))
    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    output_lst.append(y_pred_prob)
    output_lst.append(y_test)

output_df = pd.DataFrame(output_lst)
output_df.to_csv(dpath+ 'Results/S3_ANN_Pred_Probs.csv')

finaloutput = avg_results(results_cv)
finaloutput.to_csv(dpath+ 'Results/S3_ANN.csv')

