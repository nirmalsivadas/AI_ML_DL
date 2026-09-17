import numpy as np # numpy is a library for numerical computing in Python, providing support for arrays and matrices, along with a collection of mathematical functions to operate on these data structures.
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons # import the function to create a synthetic dataset
import seaborn as sns # import the library for data visualization
from mlxtend.plotting import plot_decision_regions # import the function to plot decision regions

import tensorflow # import the TensorFlow library for deep learning
from tensorflow.keras.models import Sequential # sequential model is a linear stack of layers
from tensorflow.keras.layers import Dense # dense layer is a fully connected layer
from tensorflow.keras.layers import Dropout # dropout layer is used to prevent overfitting
from tensorflow.keras.optimizers import Adam # Adam optimizer is a popular gradient-based optimizer

X, y = make_moons(100, noise=0.25,random_state=2) # create a synthetic dataset

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# without regularization
# model1 = Sequential()

# model1.add(Dense(128,input_dim=2, activation="relu"))
# model1.add(Dense(128, activation="relu"))
# model1.add(Dense(1,activation='sigmoid'))

# adam = Adam(learning_rate=0.01)
# model1.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# history1 = model1.fit(X, y, epochs=2000, validation_split = 0.2,verbose=0)

class KerasClf:
    def __init__(self, model): self.model = model
    def predict(self, X):
        return (self.model.predict(X, verbose=0) > 0.5).astype(int).ravel()

# plot_decision_regions(X, y.astype('int'), clf=KerasClf(model1), legend=2)
# plt.xlim(-2,3)
# plt.ylim(-1.5,2)
# plt.show()

# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
# plt.show()

model2 = Sequential()

model2.add(Dense(128,input_dim=2, activation="relu",kernel_regularizer=tensorflow.keras.regularizers.l2(0.001)))
model2.add(Dense(128, activation="relu",kernel_regularizer=tensorflow.keras.regularizers.l2(0.001)))
model2.add(Dense(1,activation='sigmoid'))

adam = Adam(learning_rate=0.01)
model2.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

history2 = model2.fit(X, y, epochs=2000, validation_split = 0.2,verbose=0)

plot_decision_regions(X, y.astype('int'), clf=KerasClf(model2), legend=2)
plt.xlim(-2,3)
plt.ylim(-1.5,2)
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.show()

