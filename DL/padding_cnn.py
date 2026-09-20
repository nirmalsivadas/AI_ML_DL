import tensorflow
from tensorflow import keras
from keras.layers import Dense,Conv2D,Flatten
from keras import Sequential
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# valid padding : feature map size is not same as input image size
# same padding : feature map size is same as input image size
# model = Sequential()

# model.add(Conv2D(32,kernel_size=(3,3),padding='valid', activation='relu', input_shape=(28,28,1))) # Conv2D is 2D convolutional layer used to extract features from the input image, 32 is the number of filters, kernel_size is the size of the filter, padding is the type of padding to use, activation is the activation function to use, input_shape is the shape of the input image
# model.add(Conv2D(32,kernel_size=(3,3),padding='valid', activation='relu'))
# model.add(Conv2D(32,kernel_size=(3,3),padding='valid', activation='relu'))

# model.add(Flatten()) # Flatten layer is used to flatten the output of the previous layer into a 1D array

# model.add(Dense(128,activation='relu'))
# model.add(Dense(10,activation='softmax')) # softmax layer is used to convert the output of the previous layer into a probability distribution over the 10 possible classes

# print(model.summary())

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='same',strides=(2,2), activation='relu', input_shape=(28,28,1))) # strides makes the feature map size smaller by a factor of 2 in each dimension both horizontally and vertically because the kernel size is 3x3 and the padding is same so the feature map size is the same as the input image size
model.add(Conv2D(32,kernel_size=(3,3),padding='same',strides=(2,2), activation='relu'))
model.add(Conv2D(32,kernel_size=(3,3),padding='same',strides=(2,2), activation='relu'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

print(model.summary())