import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

df = pd.read_csv('files/train (1).csv',usecols=['GarageQual','FireplaceQu','SalePrice'])

print(df.head())

print(df.isnull().mean()*100)

df['GarageQual'].value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel('GarageQual')
plt.ylabel('Number of houses')
plt.show()

df['GarageQual'].fillna('Missing', inplace=True)

df['GarageQual'].value_counts().sort_values(ascending=False).plot.bar()
plt.xlabel('GarageQual')
plt.ylabel('Number of houses')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']),
    df['SalePrice'],
    test_size=0.2,
    random_state=42,
)

imputer = SimpleImputer(strategy='constant', fill_value='Missing')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print(imputer.statistics_)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

