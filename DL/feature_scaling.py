import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\DL\\files\\Social_Network_Ads.csv')

df = df.iloc[:,2:]
print(df.head())

# not normalized code
# sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1])

# plt.show()

# X = df.iloc[:,0:2]
# y = df.iloc[:,-1]

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
# model = Sequential()

# model.add(Dense(128,activation='relu',input_dim=2))
# model.add(Dense(1,activation='sigmoid'))

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100)

# plt.plot(history.history['val_accuracy'])
# plt.show()

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=2
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sns.scatterplot(x=X_train_scaled[:, 0], y=X_train_scaled[:, 1])
plt.show()

model = Sequential()

model.add(Dense(128,activation='relu',input_dim=2))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train_scaled,y_train,validation_data=(X_test_scaled,y_test),epochs=100)

plt.plot(history.history['val_accuracy'])
plt.show()


