# TATA_Steel
Predicting Machine failure patterns and deployment of trained model

The train and test datasets were fed separately. 

The entire **EDA** has been done on the training dataset to understand patterns and analysing cause of machine failure

**Preprocessing** includes:
  dropped unnecessary columns like id and Product ID
  label encoding (or a dictionary) of categorical columns
  The column names were also processed to remove ([]_<>) symbols which might hamper model training

**Modelling** utilised the preprocessed data and modelled using supervised learning models-
  import xgboost as xgb
  import lightgbm as lgb
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import GaussianNB

LGBM had best stats among all the other models and therefore it was saved using pickle or joblib

**Deployment** was done using streamlit. The following features were added:
  It was made sure that users can upload their csv or excel files
  The necessary preprocessing will be done (this was according to the dataset train- as done in the preprocessing step)
  Next the LGBM model was applied to it to predict the Machine failures.
  And the result displayed as table and available as a download file
