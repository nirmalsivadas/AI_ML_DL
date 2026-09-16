import tensorflow as tf
import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt
import warnings
from mlxtend.plotting import plot_decision_regions
from matplotlib.colors import ListedColormap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import seaborn as sns

X, y = make_circles(n_samples=100, noise=0.1, random_state=1)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.title('Sample Data')

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

model = Sequential()

model.add(Dense(256, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=600, verbose=0)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Train and Test Loss')
plt.legend()
plt.show()

class BinaryClassifier:
	def __init__(self, model):
		self.model = model

	def predict(self, X):
		return (self.model.predict(X, verbose=0).ravel() >= 0.5).astype(int)

plt.title('Decision Plot')
plot_decision_regions(X_test, y_test.ravel(), clf=BinaryClassifier(model), legend=2)
plt.show()
