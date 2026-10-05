import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import MissingIndicator,SimpleImputer

df = pd.read_csv('files/train.csv',usecols=['Age','Fare','Survived'])

print(df.head())

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.head())

si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

print(X_train_trf)

clf = LogisticRegression()

clf.fit(X_train_trf,y_train)

y_pred = clf.predict(X_test_trf)

print(accuracy_score(y_test,y_pred))

mi = MissingIndicator() # missing indicator, what this does is it will return the features that have missing values in the dataset and the missing values will be 0 and the non missing values will be 1, a new column will be added to the dataset

mi.fit(X_train)

print(mi.features_)

X_train_missing = mi.transform(X_train)

print(X_train_missing)

X_test_missing = mi.transform(X_test)
print(X_test_missing)

X_train['Age_NA'] = X_train_missing

print(X_test)

X_test['Age_NA'] = X_test_missing

print(X_train)

si = SimpleImputer()

X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)

clf = LogisticRegression()

clf.fit(X_train_trf2,y_train)

y_pred = clf.predict(X_test_trf2)

print(accuracy_score(y_test,y_pred))

si = SimpleImputer(add_indicator=True) # missing indicator class is added

X_train = si.fit_transform(X_train)

X_test = si.transform(X_test)

clf = LogisticRegression()

clf.fit(X_train_trf2,y_train)

y_pred = clf.predict(X_test_trf2)

print(accuracy_score(y_test,y_pred))