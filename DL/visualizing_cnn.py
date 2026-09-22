from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import img_to_array, load_img
from keras.utils import plot_model
from numpy import expand_dims
from pathlib import Path
import matplotlib.pyplot as pyplot

files_dir = Path(__file__).parent / 'files'

#Load the model
model = VGG16()

print(model.summary())
try:
    plot_model(model, to_file=files_dir / 'vgg16_model.png', show_shapes=True)
except ImportError:
    print('Skipping model diagram: install pydot in the active Python environment to enable it.')

for i in range(len(model.layers)):
	# check for convolutional layer
	if 'conv' not in model.layers[i].name:
		continue
	# get filter weights
	filters, biases = model.layers[i].get_weights()
	print("layer number",i,model.layers[i].name, filters.shape)

  # retrieve weights from the second hidden layer
filters , bias = model.layers[1].get_weights()

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters =6
ix=1
fig = pyplot.figure(figsize=(15,10))
for i in range(n_filters):
    # get the filters
    f = filters[:,:,:,i]
    for j in range(3):
        # subplot for 6 filters and 3 channels
        pyplot.subplot(n_filters,3,ix)
        pyplot.imshow(f[:,:,j] ,cmap='gray')
        ix+=1
#plot the filters 
pyplot.show()

model = Model(inputs=model.inputs , outputs=model.layers[1].output)

image = load_img(files_dir / 'bread.jpg', target_size=(224, 224))

# convert the image to an array
image = img_to_array(image)
# expand dimensions so that it represents a single 'sample'
image = expand_dims(image, axis=0)

image = preprocess_input(image)

#calculating features_map
features = model.predict(image)

fig = pyplot.figure(figsize=(20,15))
for i in range(1,features.shape[3]+1):

    pyplot.subplot(8,8,i)
    pyplot.imshow(features[0,:,:,i-1] , cmap='gray')
    
pyplot.show()

model2 = VGG16()
layer_index = [ 2, 5 , 9 , 13 , 17]
outputs = [model2.layers[i].output for i in layer_index]

model3 = Model( inputs= model2.inputs, outputs = outputs)
feature_map = model3.predict(image)

for i,fmap in zip(layer_index,feature_map):
    fig = pyplot.figure(figsize=(20,15))
    fig.suptitle("Layer_{}".format(i) , fontsize=20)
    columns = 8
    rows = (fmap.shape[3] + columns - 1) // columns
    for j in range(1, fmap.shape[3] + 1):

        pyplot.subplot(rows, columns, j)
        pyplot.imshow(fmap[0, :, :, j - 1], cmap='gray')
    
pyplot.show()