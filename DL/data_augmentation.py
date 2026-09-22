import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

img = image.load_img('files/cats.jpg',target_size=(200,200))

plt.imshow(img)
plt.show()

datagen = ImageDataGenerator(
  rotation_range=40, # it will rotate the image by 40 degrees
  width_shift_range=0.2, # it will shift the image horizontally by 20%
  height_shift_range=0.2, # it will shift the image vertically by 20%
  shear_range=0.2, # it will shear the image
  zoom_range=0.2, # it will zoom the image by 20% in and out
  horizontal_flip=True, # it will flip the image horizontally
  fill_mode='nearest' # it will fill the empty space with the nearest pixel
)

img = image.img_to_array(img)
input_batch = img.reshape((1,) + img.shape) # reshape the image to (1, 200, 200, 3)

i = 0
for batch in datagen.flow(input_batch,batch_size=1,save_to_dir='files/cats_aug'): # it will save the augmented images in the 'files/cats_aug' directory
  i += 1
  if i == 20:
    break