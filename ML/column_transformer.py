import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

df = pd.read_csv('files/covid_toy.csv')
print(df.head())

print(df.isnull().sum())
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['has_covid']),df['has_covid'],test_size=0.2)

# adding simple imputer to fever col
si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[['fever']])

# also the test data
X_test_fever = si.fit_transform(X_test[['fever']])
                                 
print(X_train_fever.shape)
print(X_test_fever.shape)

# Ordinalencoding -> cough
oe = OrdinalEncoder(categories=[['Mild','Strong']])
X_train_cough = oe.fit_transform(X_train[['cough']])

# also the test data
X_test_cough = oe.fit_transform(X_test[['cough']])

print(X_train_cough.shape)

# OneHotEncoding -> gender,city
ohe = OneHotEncoder(drop='first', sparse_output=False)
X_train_gender_city = ohe.fit_transform(X_train[['gender','city']])

# also the test data
X_test_gender_city = ohe.transform(X_test[['gender','city']])

print(X_train_gender_city.shape)

# Extracting Age
X_train_age = X_train.drop(columns=['gender','fever','cough','city']).values

# also the test data
X_test_age = X_test.drop(columns=['gender','fever','cough','city']).values

print(X_train_age.shape)

X_train_transformed = np.concatenate((X_train_age,X_train_fever,X_train_gender_city,X_train_cough),axis=1)
# also the test data
X_test_transformed = np.concatenate((X_test_age,X_test_fever,X_test_gender_city,X_test_cough),axis=1)

print(X_train_transformed.shape)

# Using ColumnTransformer
transformer = ColumnTransformer(transformers=[
    ('tnf1',SimpleImputer(),['fever']),
    ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
    ('tnf3',OneHotEncoder(sparse_output=False, drop='first'),['gender','city'])
],remainder='passthrough') # what we are doing is we are creating a column transformer object, which will apply the specified transformations to the specified columns of the dataframe, and will leave the rest of the columns unchanged. The remainder parameter is set to 'passthrough', which means that the rest of the columns will be left unchanged. This is better than the previous approach, because we don't have to manually concatenate the transformed columns with the rest of the columns, and we can also easily change the transformations applied to the columns, without having to change the code for concatenating the columns.

transformer.fit_transform(X_train)
transformer.transform(X_test)

print(transformer.transform(X_train).shape)
print(transformer.transform(X_test).shape)
