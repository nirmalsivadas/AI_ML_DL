import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

df = pd.read_csv(
	'C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\DL\\files\\concertriccir2.csv',
	header=None,
	names=['X', 'Y', 'class'],
)

plt.scatter(df['X'],df['Y'],c=df['class'])
plt.show()

X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values

# without batch normalization
# model = Sequential()

# model.add(Dense(2,activation='relu',input_dim=2))
# model.add(Dense(2,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# history1 = model.fit(X,y,epochs=200,validation_split=0.2)

# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
# plt.show()

# plt.plot(history1.history['accuracy'])
# plt.plot(history1.history['val_accuracy'])
# plt.show()

model = Sequential()

model.add(Dense(3,activation='relu',input_dim=2))
model.add(BatchNormalization())
model.add(Dense(2,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history2 = model.fit(X,y,epochs=200,validation_split=0.2)

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.show()

plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.show()