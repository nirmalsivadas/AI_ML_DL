import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

(X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data() # loads the MNIST dataset, which consists of handwritten digit images and their corresponding labels, into training and testing sets.x train and y_train contain the training images and labels, while X_test and y_test contain the testing images and labels. the difference between the two datasets is that the training set is used to train the model, while the testing set is used to evaluate the model's performance on unseen data. testing is important to assess how well the model generalizes to new data and to prevent overfitting, which occurs when a model performs well on the training data but poorly on unseen data. y_train and y_test are the labels for the training and testing datasets, respectively. they represent the actual digit values (0-9) corresponding to each image in the dataset while X_train and X_test are the input features, which are the pixel values of the images. each image is represented as a 28x28 array of pixel values, where each pixel value ranges from 0 to 255, indicating the intensity of the pixel (0 for black and 255 for white). the model will learn to recognize patterns in these pixel values to classify the images into their corresponding digit labels.

# print(X_train.shape)
# print(X_test.shape)

# print(y_train.shape)
# print(y_test.shape)
# print("X_train[0]:", X_train[0])

X_train = X_train / 255.0 # normalizes the pixel values of the training images by dividing each pixel value by 255.0, scaling the values to a range between 0 and 1. this normalization step is important for improving the convergence and stability of the training process, as it ensures that the input features have similar scales.
X_test = X_test / 255.0 # normalizes the pixel values of the testing images by dividing each pixel value by 255.0, scaling the values to a range between 0 and 1. this normalization step is important for improving the convergence and stability of the training process, as it ensures that the input features have similar scales.

model = Sequential() # creates a new instance of the Sequential class, which is a linear stack of layers for building a neural network model.

model.add(Flatten(input_shape=(28, 28))) # adds a Flatten layer to the model, which reshapes the input images from a 2D array of shape (28, 28) into a 1D array of shape (784,) so that it can be fed into the subsequent dense layers. this layer does not have any learnable parameters and simply rearranges the data.
model.add(Dense(128, activation='relu')) # adds a dense (fully connected) layer to the model with 128 neurons, using the ReLU activation function. this layer will learn to extract features from the flattened input images and capture complex patterns in the data.

model.add(Dense(10, activation='softmax')) # adds another dense layer to the model with 10 neurons, using the softmax activation function. this layer serves as the output layer, where each neuron corresponds to one of the 10 possible digit classes (0-9). the softmax function ensures that the output values represent probabilities that sum to 1, allowing for multi-class classification.

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # compiles the model, specifying the loss function as sparse categorical cross-entropy (appropriate for multi-class classification with integer labels) and using the Adam optimizer for training the model. the accuracy metric is also specified to monitor the model's performance during training.

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2) # trains the model on the normalized training data (X_train) and the corresponding labels (y_train) for 10 epochs, meaning the model will iterate over the entire training dataset 10 times during training. a validation split of 0.2 is specified, meaning that 20% of the training data will be used for validation during training to monitor the model's performance on unseen data.

y_prob = model.predict(X_test) # uses the trained model to make predictions on the normalized testing data (X_test), returning the predicted probabilities for each of the 10 digit classes for each instance in the test set.

y_pred = y_prob.argmax(axis=1) # converts the predicted probabilities (y_prob) into class predictions by selecting the index of the maximum probability for each instance, which corresponds to the predicted digit label (0-9). this results in a 1D array of predicted labels for the test set.

accuracy = accuracy_score(y_test, y_pred) # calculates the accuracy score of the model on the test data by comparing the true labels (y_test) with the predicted labels (y_pred). the accuracy score represents the proportion of correctly classified instances in the test set.

print(accuracy)

plt.plot(history.history['loss'], label='loss') # plots the training loss over epochs, using the loss values stored in the history object.
plt.plot(history.history['val_loss'], label='val_loss') # plots the validation loss over epochs, using the validation loss values stored in the history object.
plt.title('Loss vs Val_Loss') # sets the title of the plot to 'Loss vs Val_Loss'.
plt.xlabel('Epochs') # sets the label for the x-axis to 'Epochs'.
plt.ylabel('Loss') # sets the label for the y-axis to 'Loss'.
plt.legend() # adds a legend to the plot, indicating which line corresponds to the training loss and which line corresponds to the validation loss.
plt.show() # displays the plot, allowing for visual inspection of the training and validation loss trends over the epochs.

plt.plot(history.history['accuracy'], label='accuracy') # plots the training accuracy over epochs, using the accuracy values stored in the history object.
plt.plot(history.history['val_accuracy'], label='val_accuracy') # plots the validation accuracy over epochs, using the validation accuracy values stored in the history object.
plt.title('Accuracy vs Val_Accuracy') # sets the title of the plot to 'Accuracy vs Val_Accuracy'.
plt.xlabel('Epochs') # sets the label for the x-axis to 'Epochs'.
plt.ylabel('Accuracy') # sets the label for the y-axis to 'Accuracy'.
plt.legend() # adds a legend to the plot, indicating which line corresponds to the training accuracy and which line corresponds to the validation accuracy.
plt.show() # displays the plot, allowing for visual inspection of the training and validation accuracy trends over the epochs.

print(model.predict(X_test[0].reshape(1, 28, 28)).argmax(axis=1)) # uses the trained model to make a prediction on the first test image (X_test[0]). the image is reshaped to match the input shape expected by the model (1, 28, 28), where 1 indicates a single instance. this returns the predicted probabilities for each of the 10 digit classes for that specific image.