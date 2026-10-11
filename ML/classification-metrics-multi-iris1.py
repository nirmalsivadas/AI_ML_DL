import numpy as np
import pandas as pd
import os

# Load the Iris dataset from a reliable public URL
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
# Standardize column names to match your expected format
df.columns = [col.capitalize() if col != 'species' else 'Species' for col in df.columns]

display(df.head())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])
display(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1], df.iloc[:, -1], test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

clf1 = LogisticRegression(max_iter=200)
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy of Logistic Regression:", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Trees:", accuracy_score(y_test, y_pred2))

print("\nLogistic Regression Confusion Matrix\n")
display(pd.DataFrame(confusion_matrix(y_test, y_pred1), columns=list(range(0, 3))))

print("\nDecision Tree Confusion Matrix\n")
display(pd.DataFrame(confusion_matrix(y_test, y_pred2), columns=list(range(0, 3))))

result = pd.DataFrame()
result['Actual Label'] = y_test
result['Logistic Regression Prediction'] = y_pred1
result['Decision Tree Prediction'] = y_pred2
display(result.sample(10))

from sklearn.metrics import precision_score, recall_score
print("\nPrecision Score:", precision_score(y_test, y_pred1, average=None))
print("Recall Score:", recall_score(y_test, y_pred1, average=None))