import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# Load MNIST dataset from Keras as a local replacement for the Kaggle dataset
(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()

# Reshape images to 1D vectors (28*28 = 784 features)
X_train_flat = X_train_full.reshape(X_train_full.shape[0], -1)
X_test_flat = X_test_full.reshape(X_test_full.shape[0], -1)

# Use a subset of the dataset to keep execution time reasonable
# Logistic regression on high-dimensional data can be slow on the full dataset
X_train, _, y_train, _ = train_test_split(X_train_flat, y_train_full, train_size=5000, random_state=2, stratify=y_train_full)
X_test, _, y_test, _ = train_test_split(X_test_flat, y_test_full, train_size=1000, random_state=2, stratify=y_test_full)

# Initialize and train classifiers
clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Generate predictions (fixing the undefined variables issue)
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

# Evaluate and display metrics
print("Accuracy of Logistic Regression:", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Trees:", accuracy_score(y_test, y_pred2))

print("\nLogistic Regression Confusion Matrix:")
display(pd.DataFrame(confusion_matrix(y_test, y_pred1), columns=list(range(10))))

print("\nDecision Tree Confusion Matrix:")
display(pd.DataFrame(confusion_matrix(y_test, y_pred2), columns=list(range(10))))

print("\nPrecision Score (Logistic Regression):", precision_score(y_test, y_pred1, average='weighted'))
print("Recall Score (Logistic Regression):", recall_score(y_test, y_pred1, average='weighted'))
print("F1 Score (Logistic Regression):", f1_score(y_test, y_pred1, average='weighted'))

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred1))