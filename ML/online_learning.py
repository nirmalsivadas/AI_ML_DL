import numpy as np
from sklearn import linear_model
import time

n_samples, n_features = 32, 500 # number of samples and features
y = np.random.randint(0, 2, size=n_samples) # target values
X = np.random.randn(n_samples, n_features) # generate random input data with a normal distribution

clf = linear_model.SGDRegressor() # create a linear regression model

start_time = time.time() # record the start time
clf.partial_fit(X, y) # partially fit the model to the input data and target values
elapsed_time = time.time() - start_time # calculate the elapsed time
print(f"Time taken for partial_fit with {n_samples} samples and {n_features} features: {elapsed_time:.6f} seconds") # print the elapsed time

# 5. Next stream block: generate another batch of 32 samples
x_next = np.random.randn(n_samples, n_features) 
y_next = np.random.randint(0, 2, size=n_samples) 

# 6. Update the model with the new mini-batch
clf.partial_fit(x_next, y_next)