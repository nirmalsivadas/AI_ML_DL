import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\DL\\files\\Admission_Predict_Ver1.1.csv')

df.drop(columns=['Serial No.'], inplace=True)

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

scalar = MinMaxScaler() # creates an instance of the MinMaxScaler class, which scales features to a given range, typically between 0 and 1. 

X_train_scaled = scalar.fit_transform(X_train) # fits the MinMaxScaler to the training data and transforms it, scaling the features in X_train to the specified range. this ensures that all features have similar scales, which can improve the performance of machine learning algorithms.
X_test_scaled = scalar.transform(X_test) # transforms the test data using the same scaling parameters learned from the training data, ensuring that the test data is scaled in the same way as the training data.

model = Sequential() # creates a new instance of the Sequential class, which is a linear stack of layers for building a neural network model.

model.add(Dense(7, activation='relu', input_dim = 7)) # adds a dense (fully connected) layer to the model with 7 neurons, using the ReLU activation function, and specifying that the input dimension is 7 (the number of features in the dataset).
model.add(Dense(7, activation='relu')) # adds another dense layer to the model with 7 neurons, using the ReLU activation function. this layer will learn to extract features from the previous layer and capture complex patterns in the data.

model.add(Dense(1, activation='linear')) # adds another dense layer to the model with 1 neuron, using the linear activation function. this layer serves as the output layer, where the model will predict a continuous value (the admission chance).

model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mean_squared_error']) # compiles the model, specifying the loss function as mean squared error (appropriate for regression tasks) and using the Adam optimizer for training the model. the mean squared error metric is also specified to monitor the model's performance during training.

history = model.fit(X_train_scaled, y_train, epochs=100,validation_split=0.2) # trains the model on the standardized training data (X_train_scaled) and the corresponding labels (y_train) for 100 epochs, meaning the model will iterate over the entire training dataset 100 times during training. a validation split of 0.2 is specified, meaning that 20% of the training data will be used for validation during training to monitor the model's performance on unseen data.

y_pred = model.predict(X_test_scaled) # uses the trained model to make predictions on the standardized test data (X_test_scaled), returning the predicted continuous values for each instance in the test set.

r2_score(y_test, y_pred)

plt.plot(history.history['loss'], label='loss') # plots the training loss over epochs, using the loss values stored in the history object.
plt.plot(history.history['val_loss'], label='val_loss') # plots the validation loss over epochs, using the validation loss values stored in the history object.
plt.title('Loss vs Val_Loss') # sets the title of the plot to 'Loss vs Val_Loss'.
plt.xlabel('Epochs') # sets the label for the x-axis to 'Epochs'.
plt.ylabel('Loss') # sets the label for the y-axis to 'Loss'.
plt.legend() # adds a legend to the plot, indicating which line corresponds to the training loss and which line corresponds to the validation loss.
plt.show() # displays the plot, allowing for visual inspection of the training and validation loss trends