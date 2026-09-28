from keras.models import Model
from keras.layers import *
from keras.utils import plot_model

x = Input(shape=(3,))
hidden1 = Dense(128,activation='relu')(x)
hidden2 = Dense(64,activation='relu')(hidden1)

output1 = Dense(1,activation='linear')(hidden2)
output2 = Dense(1,activation='sigmoid')(hidden2)

model = Model(inputs = x ,outputs = [output1,output2])
plot_model(model,show_shapes=True)



