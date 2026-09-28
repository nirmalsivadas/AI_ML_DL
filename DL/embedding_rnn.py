import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Embedding,Flatten

docs = ['go india',
		'india india',
		'hip hip hurray',
		'jeetega bhai jeetega india jeetega',
		'bharat mata ki jai',
		'kohli kohli',
		'sachin sachin',
		'dhoni dhoni',
		'modi ji ki jai',
		'inquilab zindabad']

tokenizer = Tokenizer()

tokenizer.fit_on_texts(docs) # fit the tokenizer on the documents meaning it will learn the vocabulary of the documents and store it in the tokenizer object

sequences = tokenizer.texts_to_sequences(docs)
print(sequences)

sequences = pad_sequences(sequences,padding='post')
print(sequences)

model = Sequential()

model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=2))

print(model.summary())

pred = model.predict(sequences)
print(pred)

(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=10000)
X_train = pad_sequences(X_train,padding='post',maxlen=50)
X_test = pad_sequences(X_test,padding='post',maxlen=50)

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=2))
model.add(SimpleRNN(32,return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train,epochs=5,validation_data=(X_test,y_test))