import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\DL\\files\\Churn_Modelling.csv')

# df.info() displays the summary of the DataFrame, including the number of non-null entries and data types for each column.

# df.duplicated().sum() checks for duplicate rows in the DataFrame and returns the count of duplicate rows.

# df.drop_duplicates() removes duplicate rows from the DataFrame, keeping only the first occurrence of each row.

# df.isnull() checks for missing values in the DataFrame and returns a boolean DataFrame indicating which entries are missing.

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True) # performs one-hot encoding on the 'Geography' and 'Gender' columns meaning it converts categorical variables into a format that can be provided to ML algorithms to do a better job in prediction.

X = df.drop(columns=['Exited']) # creates a new DataFrame X that contains all columns from df except for the 'Exited' column, which is the target variable for prediction.
y = df['Exited'] # creates a new Series y that contains only the 'Exited' column from df, which is the target variable for prediction.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1) # splits the dataset into training and testing sets, with 20% of the data reserved for testing and a random seed of 1 for reproducibility.

scalar = StandardScaler() # creates an instance of the StandardScaler class, which standardizes features by removing the mean and scaling to unit variance.

X_train_scaled = scalar.fit_transform(X_train) # fits the StandardScaler to the training data and transforms it, standardizing the features in X_train.
X_test_scaled = scalar.transform(X_test) # transforms the test data using the same scaling parameters learned from the training data, ensuring that the test data is standardized in the same way as the training data.

print(X_train_scaled) # prints the standardized training data to the console.

model = Sequential() # creates a new instance of the Sequential class, which is a linear stack of layers for building a neural network model.

model.add(Dense(11, activation='relu', input_dim = 11)) # adds a dense (fully connected) layer to the model with 11 neurons, using the ReLU activation function, and specifying that the input dimension is 11 (the number of features in the dataset).
model.add(Dense(11, activation='relu')) # adds another dense layer to the model with 6 neurons, using the ReLU activation function. hidden layer.



model.add(Dense(1, activation='sigmoid')) # adds another dense layer to the model with 1 neuron, using the sigmoid activation function, which is suitable for binary classification tasks. output layer.

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) # compiles the model, specifying the loss function as binary cross-entropy (appropriate for binary classification) and using the Adam optimizer for training the model.

history = model.fit(X_train_scaled, y_train, epochs=100,validation_split=0.2) # trains the model on the standardized training data (X_train_scaled) and the corresponding labels (y_train) for 100 epochs, meaning the model will iterate over the entire training dataset 100 times during training.

# model[0].get_weights() # retrieves the weights and biases of the first layer of the model, which can be useful for understanding how the model has learned to represent the input features.

y_log = model.predict(X_test_scaled) # uses the trained model to make predictions on the standardized test data (X_test_scaled), returning the predicted probabilities of the positive class (Exited = 1) for each instance in the test set.

y_pred = np.where(y_log > 0.5, 1, 0) # converts the predicted probabilities (y_log) into binary class predictions (y_pred) by applying a threshold of 0.5, where values greater than 0.5 are classified as 1 (Exited) and values less than or equal to 0.5 are classified as 0 (Not Exited).

print(accuracy_score(y_test, y_pred)) # prints the accuracy score of the model on the test data, which is calculated using accuracy_score(y_test, y_pred)

plt.plot(history.history['loss'], label='loss') # plots the training loss over epochs, using the loss values stored in the history object.
plt.plot(history.history['val_loss'], label='val_loss') # plots the validation loss over epochs, using the validation loss values stored in the history object.
plt.title('Loss vs Val_Loss') # sets the title of the plot to 'Loss vs Val_Loss'.
plt.xlabel('Epochs') # sets the label for the x-axis to 'Epochs'.
plt.ylabel('Loss') # sets the label for the y-axis to 'Loss'.
plt.legend() # adds a legend to the plot, which helps identify the different lines (loss and val_loss) in the graph.
plt.show() # displays the plot on the screen.

plt.plot(history.history['accuracy'], label='accuracy') # plots the training accuracy over epochs, using the accuracy values stored in the history object.
plt.plot(history.history['val_accuracy'], label='val_accuracy') # plots the validation accuracy over epochs, using the validation accuracy values stored in the history object.
plt.title('Accuracy vs Val_Accuracy') # sets the title of the plot to 'Accuracy vs Val_Accuracy'.
plt.xlabel('Epochs') # sets the label for the x-axis to 'Epochs'.
plt.ylabel('Accuracy') # sets the label for the y-axis to 'Accuracy'.
plt.savefig('training_accuracy.png') # saves the training accuracy plot to a file.
plt.legend() # adds a legend to the plot, which helps identify the different lines (accuracy and val_accuracy) in the graph.
plt.show() # displays the plot on the screen.