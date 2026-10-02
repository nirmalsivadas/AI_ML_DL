from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.compose import ColumnTransformer

base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / 'files' / 'train.csv')[['Age','Fare','SibSp','Parch','Survived']]

df.dropna(inplace=True)

df['family'] = df['SibSp'] + df['Parch']

df.drop(columns=['SibSp','Parch'],inplace=True) # drop SibSp and Parch columns

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Without binarization

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy without binarization",accuracy_score(y_test,y_pred))

print("Cross-Validation Accuracy without binarization",np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy')))

# With binarization

trf = ColumnTransformer([
    ('bin',Binarizer(copy=False),['family'])
],remainder='passthrough')

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)

pd.DataFrame(X_train_trf,columns=['family','Age','Fare'])

clf = DecisionTreeClassifier()
clf.fit(X_train_trf,y_train)
y_pred2 = clf.predict(X_test_trf)

print("Accuracy with binarization",accuracy_score(y_test,y_pred2))
X_trf = trf.fit_transform(X)
print("Cross-Validation Accuracy with binarization",np.mean(cross_val_score(DecisionTreeClassifier(),X_trf,y,cv=10,scoring='accuracy')))
