import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\ML\\files\\placement.csv') # read the dataset

# print(df.head()) # first 5 rows of the dataset

df = df.iloc[:,1:]

X = df.iloc[:,0:2] # input features (first two columns)
y = df.iloc[:,-1] # target variable (last column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # split the data into training and testing sets

scaler = StandardScaler() # create an instance of the StandardScaler class, which standardizes features by removing the mean and scaling to unit variance.

X_train_scaled = scaler.fit_transform(X_train) # fit the scaler to the training data and transform it, standardizing the features in X_train.
X_test_scaled = scaler.transform(X_test) # transform the test data using the same scaling parameters

clf = LogisticRegression() # create an instance of the LogisticRegression class, which implements logistic regression for binary classification tasks.

clf.fit(X_train_scaled, y_train) # fit the logistic regression model to the standardized training data and corresponding labels.

y_pred = clf.predict(X_test_scaled) # use the trained model to make predictions on the standardized test data, returning the predicted class labels for each instance in the test set.

accuracy = accuracy_score(y_test, y_pred) # calculate the accuracy of the model by comparing the predicted labels with the true labels in the test set.

plot_decision_regions(X_train_scaled, y_train.values, clf=clf, legend=2) # plot the decision regions of the trained logistic regression model using the standardized training data and corresponding labels.
plt.show()

pickle.dump(clf, open('C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\ML\\files\\placement_model.pkl', 'wb')) # save the trained logistic regression model to a file using pickle for later use.