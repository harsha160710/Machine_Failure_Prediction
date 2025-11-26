import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,recall_score

import joblib
import os

from huggingface_hub import login,HfApi,create_repo
from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError

api=HfApi(token=os.getenv('HF_TOKEN'))

xtrain_path="hf://datasets/Harsha1001/Machine-Failure-Prediction/xtrain.csv"
xtest_path="hf://datasets/Harsha1001/Machine-Failure-Prediction/xtest.csv"
ytrain_path="hf://datasets/Harsha1001/Machine-Failure-Prediction/ytrain.csv"
ytest_path="hf://datasets/Harsha1001/Machine-Failure-Prediction/ytest.csv"

x_train=pd.read_csv(xtrain_path)
x_test=pd.read_csv(xtest_path)
y_train=pd.read_csv(ytrain_path)
y_test=pd.read_csv(ytest_path)

numeric_features = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear'
]
categorical_features = ['Type']


# Class weight to handle imbalance
#class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

model_pipeline=make_pipeline(preprocessor,xgb_model)

grid_search=GridSearchCV(model_pipeline,param_grid,cv=5,scoring='recall',n_jobs=1)
grid_search.fit(x_train,y_train)

best_model=grid_search.best_estimator_

