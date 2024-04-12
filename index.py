import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

dfs = [pd.read_csv('user_' + user + '.csv') for user in ['a','b','c','d']]

for i in range(len(dfs)):
  dfs[i]['User'] = pd.Series(i, index=dfs[i].index)

data = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)

def onehot_encode(df, column):
  df=df.copy()
  dummies=pd.get_dummies(df[column], prefix=column)
  df=pd.concat([df, dummies], axis=1)
  df=df.drop(column, axis=1)
  return df

def preprocess_inputs(df, target='Class'):
  df=df.copy()
  
  targets = ['Class', 'User']
  targets.remove(target)
  df = onehot_encode(df, column=targets[0])
  
  Y=df[target].copy()
  X=df.drop(target, axis=1)
  
  X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=123)
  scaler=StandardScaler()
  scaler.fit(X_train)
  
  X_train=pd.DataFrame(scaler.transform(X_train), columns=X.columns)
  X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
  return X_train, X_test, y_train, y_test
