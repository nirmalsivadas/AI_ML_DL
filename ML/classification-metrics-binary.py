import numpy as np
import pandas as pd

# Loading the UCI Heart Disease dataset directly from the official UCI repository archive
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'

# The dataset does not have column headers, so we define them based on the UCI documentation
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

try:
    df = pd.read_csv(url, names=columns, na_values='?')
    print("Dataset loaded successfully from UCI Repository!")
    display(df.head())
except Exception as e:
    print(f"Could not load from URL, error: {e}")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Impute missing values (NaN) using the median of each column
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Split features and target
X = df_imputed.drop(columns=['target'])
y = df_imputed['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Scale features to resolve convergence issues and improve accuracy
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
clf1 = LogisticRegression(max_iter=1000)
clf2 = DecisionTreeClassifier()

clf1.fit(X_train_scaled, y_train)
clf2.fit(X_train, y_train)  # Decision Trees do not require scaling

y_pred1 = clf1.predict(X_test_scaled)
y_pred2 = clf2.predict(X_test)

print("Accuracy of Logistic Regression (Scaled):", accuracy_score(y_test, y_pred1))
print("Accuracy of Decision Trees:", accuracy_score(y_test, y_pred2))

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Get the sorted unique class labels from the test target
classes = sorted(list(y_test.unique()))

print("Logistic Regression Confusion Matrix\n")
cm1 = confusion_matrix(y_test, y_pred1, labels=classes)
display(pd.DataFrame(cm1, index=classes, columns=classes))

print("\nDecision Tree Confusion Matrix\n")
cm2 = confusion_matrix(y_test, y_pred2, labels=classes)
display(pd.DataFrame(cm2, index=classes, columns=classes))

result = pd.DataFrame()
result['Actual Label'] = y_test
result['Logistic Regression Prediction'] = y_pred1
result['Decision Tree Prediction'] = y_pred2
display(result.sample(10))

print("\nFor Logistic Regression Model")
print("-"*50)
print("Precision (per class):", precision_score(y_test, y_pred1, average=None, zero_division=0))
print("Recall (per class):", recall_score(y_test, y_pred1, average=None, zero_division=0))
print("F1 score (per class):", f1_score(y_test, y_pred1, average=None, zero_division=0))
print("Macro F1 Score:", f1_score(y_test, y_pred1, average='macro', zero_division=0))

print("\nFor DT Model")
print("-"*50)
print("Precision (per class):", precision_score(y_test, y_pred2, average=None, zero_division=0))
print("Recall (per class):", recall_score(y_test, y_pred2, average=None, zero_division=0))
print("F1 score (per class):", f1_score(y_test, y_pred2, average=None, zero_division=0))
print("Macro F1 Score:", f1_score(y_test, y_pred2, average='macro', zero_division=0))