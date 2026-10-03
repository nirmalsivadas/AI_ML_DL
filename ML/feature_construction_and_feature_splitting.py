from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent / 'files' / 'train.csv'

df = pd.read_csv(DATA_PATH)[['Age', 'Pclass', 'SibSp', 'Parch', 'Survived']]

print(df.head())

df.dropna(inplace=True)

print(df.head())

X = df.iloc[:, 0:4].copy()
y = df.iloc[:, -1]

print(X.head())

print(np.mean(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=20)))

X['Family_size'] = X['SibSp'] + X['Parch'] + 1

print(X.head())

def myfunc(num):
    if num == 1:
        # alone
        return 0
    elif num > 1 and num <= 4:
        # small family
        return 1
    else:
        # large family
        return 2

print(myfunc(4))

X['Family_type'] = X['Family_size'].apply(myfunc)

print(X.head())

X.drop(columns=['SibSp', 'Parch', 'Family_size'], inplace=True)

print(X.head())

print(np.mean(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=20)))

df = pd.read_csv(DATA_PATH)

print(df.head())

print(df['Name'])

df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

print(df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0])

print(df[['Title', 'Name']])

print(df.groupby('Title')['Survived'].mean().sort_values(ascending=False))

df['Is_Married'] = 0
df.loc[df['Title'] == 'Mrs', 'Is_Married'] = 1

print(df['Is_Married'])

