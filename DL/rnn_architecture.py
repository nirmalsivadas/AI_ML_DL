from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(3, input_shape=(4, 5)))
model.add(Dense(1, activation="sigmoid"))

print(model.summary())