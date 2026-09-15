import numpy as np
import pandas as pd

df = pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]], columns=['cgpa','profile_score','lpa'])

def initialize_parameters(layers_dims): # initialize the parameters, layers_dims is a list containing the dimensions of each layer in the network from input to output layer respectively eg [2,2,1] for a 2 input layer, 2 hidden layers and 1 output layer
    np.random.seed(3)  # set the random seed to 3 for reproducibility, random seed is used to generate random numbers
    parameters = {}  # create an empty dictionary to store the parameters
    L = len(layers_dims)  # number of layers in the network is equal to the length of the layers_dims list

    for l in range(1,L):
        parameters['W' + str(l)] = np.ones((layers_dims[l-1],layers_dims[l])) * 0.01  # initialize the weight matrix for layer l with random values from a normal distribution with mean 0 and standard deviation 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))  # initialize the bias vector for layer l with zeros

    return parameters

initialize_parameters([2,2,1])

def linear_forward(A_prev,W,b): # calculates the given neuron's linear combination of inputs
    Z = np.dot(W.T,A_prev) + b
    return Z

def L_layer_forward(X, parameters): # forward propagation
    A = X # initialize the activation of the first layer to the input data
    L = len(parameters) / 2  # number of layers in the network is equal to the number of parameters divided by 2
    for l in range(1, int(L)):
        A_prev = A
        W1 = parameters['W' + str(l)]
        b1 = parameters['b' + str(l)]
        A = linear_forward(A_prev, W1, b1)
    return A,A_prev

X = df[['cgpa','profile_score']].values[0].reshape(2,1) # first row of the dataset as input
y = df['lpa'].values[0]
parameters = initialize_parameters([2,2,1])
y_hat,A1 = L_layer_forward(X, parameters)
y_hat = y_hat[0][0]

def update_parameters(parameters,y,y_hat,A1,X):
  parameters['W2'][0][0] = parameters['W2'][0][0] + (0.001 * 2 * (y - y_hat)*A1[0][0])
  parameters['W2'][1][0] = parameters['W2'][1][0] + (0.001 * 2 * (y - y_hat)*A1[1][0])
  parameters['b2'][0][0] = parameters['W2'][1][0] + (0.001 * 2 * (y - y_hat))

  parameters['W1'][0][0] = parameters['W1'][0][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0]*X[0][0])
  parameters['W1'][0][1] = parameters['W1'][0][1] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0]*X[1][0])
  parameters['b1'][0][0] = parameters['b1'][0][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0])

  parameters['W1'][1][0] = parameters['W1'][1][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0]*X[0][0])
  parameters['W1'][1][1] = parameters['W1'][1][1] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0]*X[1][0])
  parameters['b1'][1][0] = parameters['b1'][1][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0])
  
  return parameters

update_parameters(parameters,y,y_hat,A1,X)

parameters = initialize_parameters([2, 2, 1])
epochs = 5

for i in range(epochs):
    Loss = []

    for j in range(df.shape[0]):
        X = df[['cgpa', 'profile_score']].values[j].reshape(2, 1)
        y = df['lpa'].values[j]

        y_hat, A1 = L_layer_forward(X, parameters)
        y_hat = y_hat[0][0]

        update_parameters(parameters, y, y_hat, A1, X)
        print(f'  Example - {j + 1}')
        print('    W1:', parameters['W1'])
        print('    b1:', parameters['b1'].flatten())
        print('    W2:', parameters['W2'])
        print('    b2:', parameters['b2'].flatten())
        Loss.append((y - y_hat) ** 2)

    print('Epoch - ', i + 1, 'Loss - ', np.array(Loss).mean())

parameters



    
