import os

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


df = sns.load_dataset('iris')

print(df.head())

encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

print(df.head())

df = df[['sepal_length', 'petal_length', 'species']]

print(df.head())

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(pd.DataFrame(confusion_matrix(y_test, y_pred)))

query = pd.DataFrame([[3.4, 2.7]], columns=['sepal_length', 'petal_length'])
print(clf.predict_proba(query))
print(clf.predict(query))

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X.values, y.values, clf, legend=2)

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('Softmax on Iris')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'softmax_iris_decision_regions.png'))
plt.close()

