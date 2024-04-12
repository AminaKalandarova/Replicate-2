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
  
  y=df[target].copy()
  X=df.drop(target, axis=1)
  
  X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=123)
  scaler=StandardScaler()
  scaler.fit(X_train)
  
  X_train=pd.DataFrame(scaler.transform(X_train), columns=X.columns)
  X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
  return X_train, X_test, y_train, y_test

def build_model(num_classes=3):
  inputs=tf.keras.Input(shape=(X_train, shape[1],))
  x=tf.keras.layers.Dense(128, activation='relu')(inputs)
  x=tf.keras.layers.Dense(128, activation='relu')(x)
  outputs=tf.keras.layers.Dense(num_classes, activation='softmax')(x)
  
  model = tf.keras.Model (inputs=inputs, outputs=outputs)
  model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentrapy',
    metrics=['accuracy']
  )
  
  return model

X_train, X_test, y_train, y_test = preprocess_inputs(data, target='Class')

class_model=build_model(num_classes=3)
class_history=class_model.fit(
  X-train,
  y-train,
  validation_split==0.2,
  batch_size=32,
  epochs=50,
  callbacks=[
    tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=3,
      restore_best_weights=True
    )
  ]
)

class_acc = class_model.evaluate(X_test, y_test, verbose=0) [1]
print('Test Accuracy (Class Model): {:.2f}%'.format(class_acc * 100))

