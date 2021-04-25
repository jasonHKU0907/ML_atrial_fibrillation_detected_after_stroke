
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from scipy.stats import ttest_ind, chi2_contingency, chisquare, t


dpath = '/public/home/ncrc50002/data/fudan/AF_Project/'
results_dpath = dpath + 'Models/XGB/Results/'
X = pd.read_csv(dpath + 'Data/DatasetAF.csv', low_memory = False)
y = X['y_AF']


af_indicator = [i for i in range(X.shape[0]) if y.iloc[i] == 1]
non_af_indicator = [i for i in range(X.shape[0]) if y.iloc[i] == 0]
X_AF = X.drop(non_af_indicator, axis = 0)
X_NAF = X.drop(af_indicator, axis = 0)


def continuous_summary(feature, df):
    nb_na = df[feature].isnull().sum()
    prop_na = np.round(nb_na / df.shape[0] * 100, 1)
    my_array = np.array(df[feature].dropna())
    my_mean = np.round(np.mean(my_array), 1)
    my_se = np.std(my_array) / np.sqrt(df.shape[0])
    lbd = np.round(my_mean - 1.96*my_se, 1)
    ubd = np.round(my_mean + 1.96*my_se, 1)
    return (my_mean, lbd, ubd, nb_na, prop_na)


def categorical_summary(feature, df):
    nb_na = df[feature].isnull().sum()
    prop_na = np.round(nb_na / df.shape[0] * 100, 1)
    my_array = np.array(df[feature].dropna())
    my_dict = dict(Counter(my_array))
    my_dict_prop = [np.round(item/ df.shape[0]*100, 1) for item in my_dict.values()]
    return (my_dict, my_dict_prop, nb_na, prop_na)


def get_p_value(var, df, y, type):
    '''
    find the p-value of input variable
    var: variabale name
    df: dataframe that contains var
    y: the response variable that we want to test
    type: type of the input variable, whether 'Continuous' or 'Categorical'
    '''
    var_col = df[var]
    try:
        if type == 'Continuous':
            var_yes = var_col[np.where(y == 1)[0]]
            var_no = var_col[np.where(y == 0)[0]]
            p_value = ttest_ind(var_yes, var_no, nan_policy = 'omit')[1]
        elif type == 'Categorical':
            my_tbl = pd.crosstab(var_col.fillna('NA'), y)
            p_value = chi2_contingency(my_tbl)[1]
        else:
            print('Please input the correct Type of your variable: Continuous or Categorical')
    except:
        p_value = np.nan
    return p_value


def get_p_value_prop(var, category, df, y):
    var_col = df[var]
    n1 = np.sum(y)
    n2 = len(y) - n1
    x1 = [1 for i in range(len(y)) if np.logical_and(y.iloc[i] == 1, var_col.iloc[i] == category)]
    x1 = np.sum(x1)
    x2 = [1 for i in range(len(y)) if np.logical_and(y.iloc[i] == 0, var_col.iloc[i] == category)]
    x2 = np.sum(x2)
    p1 = x1 / n1
    p2 = x2 / n2
    p_hat = (x1 + x2) / (n1 + n2)
    t_stat = (p1 - p2) / np.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))
    df = n1 + n2 - 2
    p_value = 2 *t.cdf(-abs(t_stat), df = df)
    return p_value


# 2nd identify categorical and continuous variables by checking how many levels in that feature
# use 7 as cutoff to classify categorical and continuous features
# then double check any misclassified features by visual inspection
features = X.columns
nb_levels = [len(set(X[f].fillna('NA'))) for f in features]
f_Categorical = [features[i] for i in range(len(features)) if nb_levels[i] <= 7]
f_Continuous= [features[i] for i in range(len(features)) if nb_levels[i] >= 7]

pvalues_Cate = [get_p_value(f, X, y, 'Categorical') for f in f_Categorical]
pvalues_Cont = [get_p_value(f, X, y, 'Continuous') for f in f_Continuous]

f_Categorical = pd.DataFrame(f_Categorical)
f_Continuous = pd.DataFrame(f_Continuous)
pvalues_Cate = pd.DataFrame(pvalues_Cate)
pvalues_Cont = pd.DataFrame(pvalues_Cont)


Cate_df = pd.concat((f_Categorical, pvalues_Cate), axis = 1)
Cate_df.columns = ['Features', 'pvalues']
Cate_df.sort_values(by = 'pvalues', inplace = True)
Cate_df.to_csv(results_dpath + 'Sorted_Categorical_pvalues.csv')

Cont_df = pd.concat((f_Continuous, pvalues_Cont), axis = 1)
Cont_df.columns = ['Features', 'pvalues']
Cont_df.sort_values(by = 'pvalues', inplace = True)
Cont_df.to_csv(results_dpath+ 'Sorted_Continuous_pvalues.csv')



f = 'IMG_IS_CII'

categorical_summary(f, X)
categorical_summary(f, X_AF)
categorical_summary(f, X_NAF)
get_p_value(f, X, y, 'Categorical')


f = 'ECG_PR'
f = 'ECG_HR'
f = 'UCC_TTE_LAD'
f = 'AGE'

continuous_summary(f, X)
continuous_summary(f, X_AF)
continuous_summary(f, X_NAF)
get_p_value(f, X, y, 'Continuous')

