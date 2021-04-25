import pickle
import re
import numpy as np
import pandas as pd
import warnings
import copy
import pyreadstat
from Utilities import *
from collections import Counter
from scipy.stats import ttest_ind, chi2_contingency, chisquare
#warnings.filterwarnings('ignore')

# Read dataset
# Data can only be released upon reasonable request
dpath = '../'
mydata, meta = pyreadstat.read_sas7bdat(dpath + 'Data/tt_fd_210128.sas7bdat')

# Patients are in two groups (Ischeamic stroke & TIA).
# In this study we remove the TIA patients

rm_obs = mydata['D_DIAG']
rm_idx = [i for i in range(mydata.shape[0]) if rm_obs[i] == 1]
X = X.drop(rm_idx, axis = 0)
X.reset_index(inplace = True)

features_full = mydata.columns.tolist()

# Remove features with that from follow-up interviews
rm_features_keywords1 = ['F3', 'F6', 'F12', 'm3', 'M03', 'm6', 'm12', 'M12', 'y1']
rm_features1 = [f for f in features_full if any(f.startswith(keyword) for keyword in rm_features_keywords1)]
#rm_feature_idx1 =[features_full.index(feature) for feature in rm_features1]


# Remove features with conlusion information leakage
rm_features_keywords2 = ['AF', 'ECG_R', '_Days', 'TOAST']
rm_features2 = [f for f in features_full if any(keyword in f for keyword in rm_features_keywords2)]

# Remove features obtained upon discharge & In-hospital medication and interventions
rm_features_keywords3 = ['D_', 'DM_', 'IM_', 'I_LIMB_', 'I_SWAL']
rm_features3 = [f for f in features_full if any(f.startswith(keyword) for keyword in rm_features_keywords3)]

# Remove features that contain leakage information or junk information (e.g. index)
rm_features4 = ['IMG_ccs', 'H_HD', 'MH_Arrhy', 'MH_Disease', 'A_TRANS', 'H_HD01', 'H_HD', 'ET_AT', 'code_n', 
                'ECG', 'ECG_PR_C', 'UCC_TTE_RWMA_DD']

# Remove features with date and time
date_keywords = ['_DT', '_D']
rm_features5 = [f for f in features_full if any(f.endswith(keyword) for keyword in date_keywords)]

# Remove variables contains objective or strings e.g. Doctors' short description etc
# These features were found by visual check from feature dictionary

def find_str_f(mydf):
    f_names = mydf.columns.tolist()
    f_str = []
    for f in f_names:
        my_lst = mydf[f].tolist()
        if any(isinstance(x, str) for x in my_lst):
            f_str.append(f)
    return f_str

rm_features6 = find_str_f(mydata)
rm_features6.remove('UCC_TTE_VST')
#rm_feature_idx6 = [features_full.index(feature) for feature in rm_features5]

# Remove featuers with missing values over 95%
thres = mydata.shape[0]*0.95
rm_features7 = [f for f in features_full if mydata[f].isna().sum()>thres]

# Get the final sub-extracted dataset
rm_list = list(set(rm_features1 + rm_features2 + rm_features3 + rm_features4 + rm_features5 + rm_features6 + rm_features7))
mydf = mydata.drop(rm_list, axis = 1)

# Fix a variable that contain '-' because of marked with range

tmp = []
for item in mydf['UCC_TTE_VST']:
    if '-' in item:
        lbd = int(item.split('-')[0])
        ubd = int(item.split('-')[1])
        tmp.append(np.mean((lbd, ubd)))
    elif item == '':
        tmp.append(np.nan)
    else:
        tmp.append(float(item))

mydf['UCC_TTE_VST'] = pd.DataFrame(tmp)

# Fix the variabe Waist that marked either by feet or centimeters

my_waist_feet = mydf['WAIST_FEET'].fillna(0)
my_waist = mydf['WAIST'].fillna(0) + my_waist_feet*100/3
mydf = mydf.drop(['WAIST_FEET', 'WAIST_UNIT'], axis =1)
mydf['WAIST'] = my_waist

# Features that use 98 or 99 as missing values, should be replaced, especially for those continuous variables

features  = mydf.columns.tolist()
f_98_99 = [f for f in features if np.logical_or((mydf[f].values == 98).any(), (mydf[f].values == 99).any())]
f_98_99_2np = [f for f in f_98_99 if len(set(mydf[f].fillna('NA'))) <= 7]
f_98_99_2np = f_98_99_2np + ['H_SMK_Y', 'H_DIAB_Y', 'H_HYPT_Y', 'H_LIPID_Y']

for f in f_98_99_2np:
    mydf[f][mydf[f].values == 98] = np.nan
    mydf[f][mydf[f].values == 99] = np.nan

features = list(mydf.columns)
f_binary = [f for f in features if len(set(mydf[f].fillna('NA'))) == 2]
mydf_binary_list = [BinaryTo01(f, mydf) for f in f_binary]
mydf_binary = pd.concat(mydf_binary_list, axis = 1)
mydf = mydf.drop(f_binary, axis =1)
mydf_base = pd.concat([mydf_binary, mydf], axis = 1)

mydf_base.reset_index(inplace = True)
mydf_base.to_csv('../Data/DatasetAF.csv')



