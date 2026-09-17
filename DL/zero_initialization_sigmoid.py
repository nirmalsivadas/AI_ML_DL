import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv(
	'C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\DL\\files\\ushape.csv',
	header=None,
	names=['X', 'Y', 'class'],
)

plt.scatter(df['X'],df['Y'],c=df['class'])
plt.show()

X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values

model = Sequential()

model.add(Dense(10,activation='sigmoid',input_dim=2))
model.add(Dense(1,activation='sigmoid'))

initial_weights = model.get_weights()
initial_weights[0] = np.zeros(model.get_weights()[0].shape)
initial_weights[1] = np.zeros(model.get_weights()[1].shape)
initial_weights[2] = np.zeros(model.get_weights()[2].shape)
initial_weights[3] = np.zeros(model.get_weights()[3].shape)

model.set_weights(initial_weights)

print(model.get_weights())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X,y,epochs=100,validation_split=0.2)


print(model.get_weights())

class DecisionRegionClassifier:
	def __init__(self, model):
		self.model = model

	def predict(self, inputs):
		probabilities = self.model.predict(inputs, verbose=0)
		return (probabilities.ravel() >= 0.5).astype(int)


plot_decision_regions(X, y.astype('int'), clf=DecisionRegionClassifier(model), legend=2)
plt.show()
