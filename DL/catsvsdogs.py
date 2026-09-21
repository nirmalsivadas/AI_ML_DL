import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import os

# Check if the data folder exists. 
# If you extracted it as instructed, it should be 'PetImages'
data_dir = "PetImages" # Change this to "files/data" if you manually created those folders

if not os.path.exists(data_dir):
    print(f"Error: {data_dir} not found. Please check your downloaded dataset folder name.")
else:
    print(f"Data found at {data_dir}")

    # Using validation_split to automatically divide data into train and validation
    train_ds = keras.utils.image_dataset_from_directory(
        directory=data_dir,
        validation_split=0.2, # 80% training, 20% validation
        subset="training",
        seed=123,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 256)
    )

    validation_ds = keras.utils.image_dataset_from_directory(
        directory=data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(256, 256)
    )

    # Normalize pixel values from [0, 255] to [0, 1]
    def process(image, label):
        image = tf.cast(image / 255., tf.float32)
        return image, label

    train_ds = train_ds.map(process)
    validation_ds = validation_ds.map(process)

    # CNN MODEL ARCHITECTURE
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

    # Plot Accuracy
    plt.plot(history.history['accuracy'], label='train', color='blue')
    plt.plot(history.history['val_accuracy'], label='validation', color='red')
    plt.legend()
    plt.show()

    # Plot Loss
    plt.plot(history.history['loss'], label='train', color='blue')
    plt.plot(history.history['val_loss'], label='validation', color='red')
    plt.legend()
    plt.show()