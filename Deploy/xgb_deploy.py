
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
import xgboost as xgb
dpath = '../'

X_deploy = pd.read_csv(dpath + 'Deploy/Input.csv', low_memory = False)

y_deploy_prob = np.reshape(np.zeros(6), (1, 6))

for i in range(5):
    my_xgb = pickle.load(open(dpath + 'Deploy/XGB_cv{}.pickle.dat'.format(i), 'rb'))
    y_deploy_prob[0, i+1] = my_xgb.predict_proba(X_deploy)[0][1]

y_deploy_prob[0, 0] = np.mean(y_deploy_prob[0, 1:])

y_deploy_prob = pd.DataFrame(y_deploy_prob, columns = ['Xgb_ensemble', 'Xgb_cv1', 'Xgb_cv2', 'Xgb_cv3', 'Xgb_cv4', 'Xgb_cv5'])
y_deploy_prob.to_csv(dpath+ 'Deploy/Output.csv')








