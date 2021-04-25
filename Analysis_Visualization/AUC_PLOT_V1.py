


import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd


dpath = '../'

pred_xgb = pd.read_csv(dpath + Results/'S3_XGB_Pred_Probs.csv')
pred_rf = pd.read_csv(dpath + 'Results/S3_RF_Pred_Probs.csv'
pred_lgr = pd.read_csv(dpath + 'Results/S3_LGR_Pred_Probs.csv')
pred_knn = pd.read_csv(dpath + 'Results/S3_KNN_Pred_Probs.csv')
pred_svm = pd.read_csv(dpath + 'Results/S3_SVM_Pred_Probs.csv')
pred_ann = pd.read_csv(dpath + 'Results/S3_ANN_Pred_Probs.csv')

plt.figure(figsize = (8, 8))
tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in [0, 2, 4, 6, 8]:
    y_pred = pred_xgb.iloc[i, 1:2830]
    y_true = pred_xgb.iloc[i+1, 1:2830]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'green', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'green', linewidth = 2, 
         label = 'XGBoost                      0.867                ')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'mediumspringgreen', alpha = 0.2)


tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in [0, 2, 4, 6, 8]:
    y_pred = pred_rf.iloc[i, 1:2829]
    y_true = pred_rf.iloc[i+1, 1:2829]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'blue', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'blue', linewidth = 2, 
         label = 'Random Forest             0.855                ')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightskyblue', alpha = 0.2)


tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in [0, 2, 4, 6, 8]:
    y_pred = pred_ann.iloc[i, 1:2829]
    y_true = pred_ann.iloc[i+1, 1:2829]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'darkorange', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'dimgrey', linewidth = 2,
         label = 'ANN                              0.838                ')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'silver', alpha = 0.2)


tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in [0, 2, 4, 6, 8]:
    y_pred = pred_svm.iloc[i, 1:2829]
    y_true = pred_svm.iloc[i+1, 1:2829]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'mediumvioletred', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'mediumvioletred', linewidth = 2, 
         label = 'SVM                              0.829                ')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'hotpink', alpha = 0.2)



tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in [0, 2, 4, 6, 8]:
    y_pred = pred_lgr.iloc[i, 1:2829]
    y_true = pred_lgr.iloc[i+1, 1:2829]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'red', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'red', linewidth = 2, 
         label = 'Logistic Regression       0.829                ')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'lightsalmon', alpha = 0.2)



tprs = []
base_fpr = np.linspace(0, 1, 101)

for i in [0, 2, 4, 6, 8]:
    y_pred = pred_knn.iloc[i, 1:2829]
    y_true = pred_knn.iloc[i+1, 1:2829]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, 'darkorange', alpha = 0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, 'darkorange', linewidth = 2,
         label = 'KNN                              0.816                ')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = 'wheat', alpha = 0.2)


plt.legend(loc = 'lower right',  fontsize = 24)
plt.plt([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.
plt.ylabel('True Positive Rate', fontsize = 24)
plt.xlabel('False Positive Rate', fontsize = 24)
plt.axes().set_aspect('equal', 'datalim')
plt.show()

