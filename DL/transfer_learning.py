import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

conv_base = VGG16( # VGG16 is a pre-trained model
    weights='imagenet', # load weights pre-trained on ImageNet
    include_top = False, # do not include the ImageNet classification layer meaning that the model will be used for feature extraction. Classification layer is the last layer of the model
    input_shape=(150,150,3) # input image size
)

model = Sequential()

model.add(conv_base) # add the pre-trained model as the first layer of the model
model.add(Flatten()) # flatten the output of the pre-trained model
model.add(Dense(256,activation='relu')) # add a dense layer with 256 units and ReLU activation
model.add(Dense(1,activation='sigmoid')) # add a dense layer with 1 unit and sigmoid activation

model.summary()

conv_base.trainable = False # freeze the pre-trained model

# for fine tuning
# conv_base.trainable = True

# set_trainable = False

# for layer in conv_base.layers:
#   if layer.name == 'block5_conv1':
#     set_trainable = True
#   if set_trainable:
#     layer.trainable = True
#   else:
#     layer.trainable = False

# for layer in conv_base.layers:
#   print(layer.name,layer.trainable)

# generators for training and validation
# train_ds and validation_ds are generators that yield batches of images and labels from the train and test directories
train_ds = keras.utils.image_dataset_from_directory( # create a dataset from a directory of images
    directory = '/content/train',  # directory of images
    labels='inferred', # infer labels from directory structure
    label_mode = 'int', # labels are integers
    batch_size=32,
    image_size=(150,150)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(150,150)
)

#data augmentation
# batch_size = 32

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         '/content/train',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary') 

# validation_generator = test_datagen.flow_from_directory(
#         '/content/test',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')

# Normalize
def process(image,label): # function to normalize the images and labels, what this does is to divide each pixel value by 255 to normalize it to the range [0, 1]. This is done to ensure that the input features have similar scales and can be processed effectively by the neural network.
    image = tensorflow.cast(image/255. ,tensorflow.float32)
    return image,label

train_ds = train_ds.map(process) 
validation_ds = validation_ds.map(process)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# model.compile(
#     optimizer=keras.optimizers.RMSprop(lr=1e-5),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
#   ) # for fine tuning

history = model.fit(train_ds,epochs=10,validation_data=validation_ds)
# history = model.fit_generator(
#         train_generator,
#         epochs=10,
#         validation_data=validation_generator) # for data augmentation

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()