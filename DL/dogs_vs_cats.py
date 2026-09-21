import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import cv2

train_dir = "files/data/train"
test_dir = "files/data/test"

# generators for training and validation are used for decreasing ram usage and faster training time and increased accuracy 
train_ds = keras.utils.image_dataset_from_directory(
  directory=train_dir,
    labels='inferred', # inferred meaning labels are inferred from the directory structure
    label_mode='int', # cats are 0, dogs are 1
    batch_size=32, # batches because of limited ram
    image_size=(256, 256) # images will be of same size as compared to input
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory=test_dir,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)
# Normalize pixel values from [0, 255] to [0, 1], what this does is to divide each pixel value by 255 to normalize it to the range [0, 1]. This is done to ensure that the input features have similar scales and can be processed effectively by the neural network.
def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# #CNN MODEL ARCHITECTURE
model = Sequential()

# 1st Convolutional Block
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# 2nd Convolutional Block
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# 3rd Convolutional Block
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Flattening and Fully Connected Layers
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))

# Output Layer for Binary Classification
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

# MODEL COMPILATION & TRAINING
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Execute training across 10 epochs
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)

plt.plot(history.history['accuracy'], label='train', color='blue')
plt.plot(history.history['val_accuracy'], label='validation', color='red')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train', color='blue')
plt.plot(history.history['val_loss'], label='validation', color='red')
plt.legend()
plt.show()