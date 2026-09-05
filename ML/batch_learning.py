import numpy as np
from sklearn import linear_model
import time

n_samples, n_features = 32, 500 # number of samples and features
y = np.random.randint(0, 2, size=n_samples) # target values
X = np.random.randn(n_samples, n_features) # generate random input data with a normal distribution

clf = linear_model.SGDRegressor() # create a linear regression model

start_time = time.time() # record the start time
clf.fit(X, y) # fit the model to the input data and target values
elapsed_time = time.time() - start_time # calculate the elapsed time
print(f"Time taken for fit with {n_samples} samples and {n_features} features: {elapsed_time:.6f} seconds") # print the elapsed time