# The application of machine learning models in identification of atrial fibrillation detected after stroke


## Dependencies
  numpy==1.18.1\
  pandas==1.1.5\
  scipy==1.4.1\
  shap==0.37.0\
  matplotlib==2.2.3\
  re=2.2.1\
  imblearn==0.7.0\
  sklearn==0.23.0\
  tensorflow==1.15.0\
  keras==2.3.1\
  xgboost==1.2.1\
  lightgbm==3.1.0
  
## Code structures
- **Preprocessing**
  
  Data screening and preprocessing
  
- **Utility**

  Customizable functions for data preprocessing, model training and evaluation
  
- **Train**
  - S1*.py
    Feature importance ranking (tree-based impurity & Shap values)
  - S2*.py
    Sequential forward feature selection
  - S3*.py
    Model construction & training
  
- **Evaluation and Visualization**

  - Feature analysis
  - Data & Model visualization
  
- **Deploy**

  This contains the pretrained weights for five xgboost models (5-fold crossvalidation).
  
  You can cusotomize the input.csv with specified values of predictors and run xgb_deploy.py to generate a output.csv file with predicted probabilities
  - *ECG PR intervel*
    - recommended range: [120 - 240] (based on quantile 0.01 - 0.99 in CNSR-III)
  - *Age*
    - recommended range: [35 - 90] (based on quantile 0.01 - 0.99 in CNSR-III)
  - *Left Atrium Diameter*
    - recommended range: [24 - 52] (based on quantile 0.01 - 0.99 in CNSR-III)
  - *ECG Heart Rate*
    - recommended range: [45 - 115] (based on quantile 0.01 - 0.99 in CNSR-III)
  - *Whether is cortical Infarct*
    - 0: no infarct observed
    - 1: infarct observed: cortical infarct
    - 2: infarct observed: not cortical infarct
  


### THE PRE-TRAINED WEIGHTS CAN ONLY BE TESTED ON THE PURPOSE OF RESEARCHES. THIS HAS NOT BEEN REVIEWED OR APPROVED BY THE FOOD AND DRUG ADMINISTRATION OR BY ANY OTHER AGENCIES. YOU SHOULD ACKNOWLEDGE AND AGREE THAT CLINICAL APPLICATIONS ARE NEITHER RECOMMENDED NOR ADVISED.

  


