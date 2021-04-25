# The application of machine learning models in identification of atrial fibrillation detected after stroke


## Dependencies


## Code structures
- Preprocessing
  Data screening and preprocessing
  
- Utitilies
  Customizable functions for data preprocessing and model evaluation
  
- Train
  S1*.py
  - Feature importance ranking (tree-based impurity & Shap values)
  S2*.py
  - Sequential forward feature selection
  S3*.py
  - Model construction & training
  
- Evaluation and Visualization
  Feature analysis
  Data & Model visualization
  
- Deploy
  This contains the pretrained weights for five xgboost models (5-fold crossvalidation)
  You can cusotomize the input.csv with specified values of predictors and run xgb_deploy.py to generate a output.csv file with predicted probabilities
  - ECG PR intervel
    - recommended range: [120 - 240] (based on quantile 0.01 - 0.99 in CNSR-III)
  - Age
    - recommended range: [35 - 90] (based on quantile 0.01 - 0.99 in CNSR-III)
  - Left Atrium Diameter
    - recommended range: [24 - 52] (based on quantile 0.01 - 0.99 in CNSR-III)
  - ECG Heart Rate
    - recommended range: [45 - 115] (based on quantile 0.01 - 0.99 in CNSR-III)
  - Whether is cortical Infarct
    - 0: no infarct observed
    - 1: infarct observed: cortical infarct
    - 2: infarct observed: not cortical infarct
  


The pre-trained weights can only been tested on the purpose of researches. This has not been reviewed or approved by the food and drug administration or by any other agency.
You acknowledge and agree that clinical applications are neither recommended nor advised.



  


