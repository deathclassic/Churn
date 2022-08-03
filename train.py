
import pandas as pd
import numpy as np

import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# parameters

C = 10
output_file = 'model_C=10.bin'

# data prep

df = pd.read_csv('data/ChurnData.csv')
df.head().T

df.dtypes

df.columns = [col.lower().replace(' ', '_') for col in df.columns]
obj_cols = list(df.select_dtypes('object'))
for col in obj_cols:
  df[col] = df[col].str.lower().str.replace(' ', '_')

df.head()

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

num = ['tenure', 'monthlycharges', 'totalcharges']
cat = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# model validation

def train(df_train, y_train, C=1.0):
  train_dict = df_train[cat + num].to_dict(orient='records')

  dv = DictVectorizer(sparse=False)
  X_train = dv.fit_transform(train_dict)

  model = LogisticRegression(C=C, max_iter=1000)
  model.fit(X_train, y_train)

  return dv, model

def predict(df, dv, model):
  val_dict = df[cat+num].to_dict(orient='records')

  X_val = dv.transform(val_dict)
  y_pred = model.predict_proba(X_val)[:, 1]

  return y_pred

scores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
  df_train = df_full_train.iloc[train_idx]
  df_val = df_full_train.iloc[val_idx]

  y_train = df_train.churn.values
  y_val = df_val.churn.values

  dv, model = train(df_train, y_train, C=C)
  y_pred = predict(df_val, dv, model)

  auc = roc_auc_score(y_val, y_pred)
  scores.append(auc)
  
print(f'C value: {C} mean auc: {np.mean(scores):.3f}\tstd: {np.std(scores):.3f}')

# final model 

y_test = df_test.churn.values

dv, model = train(df_full_train, df_full_train.churn.values, C=10)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc

# saving the model

with open(output_file, 'wb') as f_out:
  pickle.dump((dv, model), f_out)

print(f'The model is saved in {output_file}')
