
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import login,HfApi

api=HfApi(token=os.getenv('HF_TOKEN'))
dataset_path="week_2_mls/data/bank_customer_churn (1).csv"
df=pd.read_csv(dataset_path)
print("Dataset loaded successfully")

df.drop(columns=['UDI'],inplace=True)

label_encoder=LabelEncoder()
df['Type']=label_encoder.fit_transform(df['Type'])

target_col='Failure'

X=df.drop(columns=[target_col])
Y=df[target_col]

x_train,y_train,x_test,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

x_train.to_csv("xtrain.csv",index=False)
y_train.to_csv("ytrain.csv",index=False)
x_test.to_csv("xtest.csv",index=False)
y_test.to_csv("ytest.csv",index=False)

files=["xtrain.csv","ytrain.csv","xtest.csv",'ytest.csv']

for file_path in files:
  api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Harsha1001/Machine-Failure-Prediction",
        repo_type="dataset",
    )
