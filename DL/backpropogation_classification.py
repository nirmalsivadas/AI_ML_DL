import numpy as np
import pandas as pd

df = pd.DataFrame([[8, 8, 1], [7, 9, 1], [6, 10, 1], [5, 12, 0]], columns=['cgpa', 'profile_score', 'placed'])

def initialize_parameters(layers_dims): # initialize the parameters, layers_dims is a list containing the dimensions of each layer in the network from input to output layer respectively eg [2,2,1] for a 2 input layer, 2 hidden layers and 1 output layer
    np.random.seed(3)  # set the random seed to 3 for reproducibility, random seed is used to generate random numbers
    parameters = {}  # create an empty dictionary to store the parameters
    L = len(layers_dims)  # number of layers in the network is equal to the length of the layers_dims list

    for l in range(1,L):
        parameters['W' + str(l)] = np.ones((layers_dims[l-1],layers_dims[l])) * 0.01  # initialize the weight matrix for layer l with random values from a normal distribution with mean 0 and standard deviation 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))  # initialize the bias vector for layer l with zeros

    return parameters

def sigmoid(z): # sigmoid activation function
    return 1 / (1 + np.exp(-z))

initialize_parameters([2,2,1])

def linear_forward(A_prev,W,b): # calculates the given neuron's linear combination of inputs
    Z = np.dot(W.T,A_prev) + b
    A = sigmoid(Z)
    return A

def L_layer_forward(X, parameters): # forward propagation
    A = X # initialize the activation of the first layer to the input data
    L = len(parameters) // 2  # number of layers in the network is equal to the number of parameters divided by 2
    for l in range(1, L + 1):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A = linear_forward(A_prev, W, b)
    return A,A_prev

X = df[['cgpa','profile_score']].values[0].reshape(2,1) # first row of the dataset as input
y = df['placed'].values[0]
parameters = initialize_parameters([2,2,1])
y_hat,A1 = L_layer_forward(X, parameters)
y_hat = y_hat[0][0]

def update_parameters(parameters, y, y_hat, A1, X):
        learning_rate = 0.0001
        output_error = y - y_hat

        parameters['W2'][0][0] += learning_rate * output_error * A1[0][0]
        parameters['W2'][1][0] += learning_rate * output_error * A1[1][0]
        parameters['b2'][0][0] += learning_rate * output_error

        hidden_error = output_error * parameters['W2'][:, 0:1] * A1 * (1 - A1)
        parameters['W1'] += learning_rate * X @ hidden_error.T
        parameters['b1'] += learning_rate * hidden_error

        return parameters

update_parameters(parameters,y,y_hat,A1,X)

parameters = initialize_parameters([2, 2, 1])
epochs = 50

for i in range(epochs):
    Loss = []

    for j in range(df.shape[0]):
        X = df[['cgpa', 'profile_score']].values[j].reshape(2, 1)
        y = df['placed'].values[j]

        y_hat, A1 = L_layer_forward(X, parameters)
        y_hat = y_hat[0][0]

        update_parameters(parameters, y, y_hat, A1, X)
        y_hat_for_loss = np.clip(y_hat, 1e-7, 1 - 1e-7)
        Loss.append(-y * np.log(y_hat_for_loss) - (1 - y) * np.log(1 - y_hat_for_loss))

    print('Epoch - ', i + 1, 'Loss - ', np.array(Loss).mean())
    print('  W1:', parameters['W1'])
    print('  b1:', parameters['b1'].flatten())
    print('  W2:', parameters['W2'])
    print('  b2:', parameters['b2'].flatten())

parameters



    
