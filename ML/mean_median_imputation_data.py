from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / 'files' / 'titanic_toy.csv'
df = pd.read_csv(csv_path)

print(df.head())

print(df.info())

print(df.isnull().mean())

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.shape)
print(X_test.shape)

print(X_train.isnull().mean())

mean_age = X_train['Age'].mean() # calculating mean for mean imputation
median_age = X_train['Age'].median() # calculating median for median imputation

mean_fare = X_train['Fare'].mean() # calculating mean for mean imputation
median_fare = X_train['Fare'].median() # calculating median for median imputation

X_train['Age_median'] = X_train['Age'].fillna(median_age) # imputing missing values with median 
X_train['Age_mean'] = X_train['Age'].fillna(mean_age) # imputing missing values with mean

X_train['Fare_median'] = X_train['Fare'].fillna(median_fare) # imputing missing values with median
X_train['Fare_mean'] = X_train['Fare'].fillna(mean_fare) # imputing missing values with mean

print(X_train.sample(5))

print('Original Age variable variance: ', X_train['Age'].var())
print('Age Variance after median imputation: ', X_train['Age_median'].var())
print('Age Variance after mean imputation: ', X_train['Age_mean'].var())

print('Original Fare variable variance: ', X_train['Fare'].var())
print('Fare Variance after median imputation: ', X_train['Fare_median'].var())
print('Fare Variance after mean imputation: ', X_train['Fare_mean'].var())

fig = plt.figure()
ax = fig.add_subplot(111)

# original variable distribution
X_train['Age'].plot(kind='kde', ax=ax)

# variable imputed with the median
X_train['Age_median'].plot(kind='kde', ax=ax, color='red')

# variable imputed with the mean
X_train['Age_mean'].plot(kind='kde', ax=ax, color='green')

# add legends
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

# original variable distribution
X_train['Fare'].plot(kind='kde', ax=ax)

# variable imputed with the median
X_train['Fare_median'].plot(kind='kde', ax=ax, color='red')

# variable imputed with the mean
X_train['Fare_mean'].plot(kind='kde', ax=ax, color='green')

# add legends
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.show()

print(X_train.cov())

print(X_train.corr())

X_train[['Age', 'Age_median', 'Age_mean']].boxplot()
plt.show()

X_train[['Fare', 'Fare_median', 'Fare_mean']].boxplot()
plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

imputer1 = SimpleImputer(strategy='median')
imputer2 = SimpleImputer(strategy='mean')

trf = ColumnTransformer([
    ('imputer1',imputer1,['Age']),
    ('imputer2',imputer2,['Fare'])
],remainder='passthrough')

trf.fit(X_train)

print(trf.named_transformers_['imputer1'].statistics_)

print(trf.named_transformers_['imputer2'].statistics_)

X_train = trf.transform(X_train)
X_test = trf.transform(X_test)

print(X_train)

