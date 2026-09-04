import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt # data visualization library
from sklearn.linear_model import Perceptron # import the perceptron model
from mlxtend.plotting import plot_decision_regions # import the function to plot decision regions

df = pd.read_csv('Single Layer Perceptron Dataset.csv').dropna(how='all') # read the dataset and drop any rows that are all NaN

print(df.shape) # how many rows and columns are in the dataset
df.head() # first 5 rows of the dataset
sns.scatterplot(data=df, x='Feature2',y='Feature3', hue='Class_Label', palette='bright') # scatter plot of the dataset, color coded by class label, palette is bright, data is df, x is Feature2, y is Feature3, hue is Class_Label, Feature1 is the independent variable, Feature2 and Feature3 are the dependent variables
# plt.show() # show the plot

x = df.iloc[:, 2:4].values # independent variables, all rows, columns 2 and 3 (Feature2 and Feature3)
y = df.iloc[:, 4].values.astype(int) # dependent variable, all rows, column 4 (Class_Label)
p = Perceptron()

p.fit(x, y) # fit the model to the data

# print(p.coef_) # weights of the model

# print(p.intercept_) # bias of the model

# print(p.score(x, y)) # accuracy of the model on the training data

plot_decision_regions(x, y, clf=p, legend=2) # plot the decision regions of the model on the training data, x is the independent variables, y is the dependent variable, clf is the model, legend is the number of classes

plt.xlabel('Feature2') # label for the x axis
plt.ylabel('Feature3') # label for the y axis
plt.show() # show the plot

