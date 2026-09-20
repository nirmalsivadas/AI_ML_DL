import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
import keras_tuner as kt

df = pd.read_csv('C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\DL\\files\\diabetes.csv')

# print(df.head())

# print(df.corr()) # correlation matrix

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

scalar = StandardScaler()
X = scalar.fit_transform(X) # standardizing the features to have zero mean and unit variance

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

model = Sequential()

model.add(Dense(32,activation='relu',input_dim=8))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test))

