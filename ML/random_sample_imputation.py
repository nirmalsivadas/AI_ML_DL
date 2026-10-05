import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('files/train.csv', usecols=['Age', 'Fare', 'Survived'])

print(df.head())
print(df.isnull().mean() * 100)

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train = X_train.copy()
X_test = X_test.copy()

X_train['Age_imputed'] = X_train['Age']
X_test['Age_imputed'] = X_test['Age']

missing_train = X_train['Age_imputed'].isnull()
available_train_ages = X_train['Age'].dropna()
for index, observation in X_train.loc[missing_train].iterrows():
    sampled_value = available_train_ages.sample(
        1, random_state=int(observation['Fare'])
    ).iloc[0]
    X_train.loc[index, 'Age_imputed'] = sampled_value

missing_test = X_test['Age_imputed'].isnull()
for index, observation in X_test.loc[missing_test].iterrows():
    sampled_value = available_train_ages.sample(
        1, random_state=int(observation['Fare'])
    ).iloc[0]
    X_test.loc[index, 'Age_imputed'] = sampled_value

print(X_train)
print('Age missing in training set:', X_train['Age'].isnull().sum())

sns.kdeplot(X_train['Age'], label='Original', fill=False)
sns.kdeplot(X_train['Age_imputed'], label='Imputed', fill=False)
plt.legend()
plt.show()

print('Original variable variance: ', X_train['Age'].var())
print('Variance after random imputation: ', X_train['Age_imputed'].var())
print(X_train[['Fare', 'Age', 'Age_imputed']].cov())

X_train[['Age', 'Age_imputed']].boxplot()
plt.show()

data = pd.read_csv('files/house-train.csv',usecols=['GarageQual','FireplaceQu', 'SalePrice'])

print(data.head())
print(data.isnull().mean() * 100)

X = data
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train = X_train.copy()
X_test = X_test.copy()

X_train['GarageQual_imputed'] = X_train['GarageQual']
X_test['GarageQual_imputed'] = X_test['GarageQual']
X_train['FireplaceQu_imputed'] = X_train['FireplaceQu']
X_test['FireplaceQu_imputed'] = X_test['FireplaceQu']

print(X_train.sample(5))

missing_garage_train = X_train['GarageQual_imputed'].isnull()
if missing_garage_train.sum() > 0:
    X_train.loc[missing_garage_train, 'GarageQual_imputed'] = X_train['GarageQual'].dropna().sample(missing_garage_train.sum()).values # fill with random values from the training set without missing values in the variable

missing_garage_test = X_test['GarageQual_imputed'].isnull()
if missing_garage_test.sum() > 0:
    X_test.loc[missing_garage_test, 'GarageQual_imputed'] = X_train['GarageQual'].dropna().sample(missing_garage_test.sum()).values

missing_fireplace_train = X_train['FireplaceQu_imputed'].isnull()
if missing_fireplace_train.sum() > 0:
    X_train.loc[missing_fireplace_train, 'FireplaceQu_imputed'] = X_train['FireplaceQu'].dropna().sample(missing_fireplace_train.sum()).values

missing_fireplace_test = X_test['FireplaceQu_imputed'].isnull()
if missing_fireplace_test.sum() > 0:
    X_test.loc[missing_fireplace_test, 'FireplaceQu_imputed'] = X_train['FireplaceQu'].dropna().sample(missing_fireplace_test.sum()).values

temp = pd.concat(
    [
        X_train['GarageQual'].value_counts() / len(X_train['GarageQual'].dropna()),
        X_train['GarageQual_imputed'].value_counts() / len(X_train)
    ],
    axis=1,
)

temp.columns = ['original', 'imputed']
print(temp)

temp = pd.concat(
    [
        X_train['FireplaceQu'].value_counts() / len(X_train['FireplaceQu'].dropna()),
        X_train['FireplaceQu_imputed'].value_counts() / len(X_train)
    ],
    axis=1,
)

temp.columns = ['original', 'imputed']
print(temp)

for category in X_train['FireplaceQu'].dropna().unique():
    sns.kdeplot(X_train.loc[X_train['FireplaceQu'] == category, 'SalePrice'], label=str(category), fill=False)
plt.show()

for category in X_train['FireplaceQu_imputed'].dropna().unique():
    sns.kdeplot(X_train.loc[X_train['FireplaceQu_imputed'] == category, 'SalePrice'], label=str(category), fill=False)
plt.show()