import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt # data visualization library
from sklearn.datasets import make_classification # import the function to create a synthetic dataset
from mlxtend.plotting import plot_decision_regions # import the function to plot decision regions

df = pd.read_csv(r'C:\Users\Nirmal\OneDrive\Desktop\AI_ML_DL\DL\files\Single Layer Perceptron Dataset.csv').dropna(how='all') # read the dataset and drop any rows that are all NaN

x,y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42) # create a synthetic dataset with 100 samples, 2 features, 2 informative features, and 0 redundant features, random state is 42

# plt.figure(figsize=(10, 6)) # set the figure size to 10x6 inches
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap='winter', edgecolor='k',s=100) # scatter plot of the synthetic dataset, color coded by class label, cmap is winter, edgecolor is black and s is the size of the points

# plt.xlabel('Feature 1') # label for the x axis
# plt.ylabel('Feature 2') # label for the y axis
# plt.show() # show the plot


def step(z):
  return 1 if z>0 else 0 # return 1 if z is greater than 0, else return 0

def perceptron(x,y,):
  x = np.insert(x,0,1,axis=1) # add a bias term to the input data, insert a column of 1s at index 0 along axis 1
  weights = np.ones(x.shape[1]) # initialize the weights to 1s, shape is the number of features in the input data
  lr = 0.1 # learning rate is 0.1
  for i in range(1000): # for 1000 iterations
    j = np.random.randint(0,x.shape[0]) # randomly select an index from the input data
    y_hat = step(np.dot(x[j],weights)) # calculate the dot product of the input data and the weights
    weights = weights + lr*(y[j]-y_hat)*x[j] # update the weights using the perceptron learning rule
  return weights[0],weights[1:] # return the weights after 1000 iterations

intercept_,coef_= perceptron(x,y) # call the perceptron function and get the intercept and coefficients
print(intercept_,coef_) # print the intercept and coefficients

m = -(coef_[0]/coef_[1]) # calculate the slope of the decision boundary
b = -(intercept_/coef_[1]) # calculate the y-intercept of the decision

x_input = np.linspace(-3,3,100) # create an array of 100 points between -3 and 3 for the x-axis
y_input = m*x_input + b # calculate the corresponding y values for the decision boundary
plt.plot(x_input,y_input,color='red',linewidth=3) # plot the decision boundary
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='winter', edgecolor='k',s=100) # scatter plot of the synthetic dataset, color coded by class label, cmap is winter, edgecolor is black and s is the size of the points
plt.ylim(-3,2) # set the y limits of the plot to -3 and 2
plt.xlim(-3,3) # set the x limits of the plot to -3 and 3
plt.xlabel('Feature 1') # label for the x axis
plt.ylabel('Feature 2') # label for the y axis
plt.show() # show the plot