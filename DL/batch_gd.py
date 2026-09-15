import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).parent / 'files' / 'Social_Network_Ads.csv')

print(df.head())

df = df[['Age','EstimatedSalary','Purchased']]

print(df.head())

X = df.iloc[:,0:2]
y = df.iloc[:,-1]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = Sequential()

model.add(Dense(10,activation='relu',input_dim=2))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',metrics=['accuracy'])
#start = time.time()
history = model.fit(X_scaled,y,epochs=10,batch_size=250,validation_split=0.2)
#print(time.time() - start)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs Val_Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

