import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('files/customer.csv')

print(df.sample())
df = df.iloc[:, 2:]

print(df.head())

X = df.drop('purchased', axis=1)
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oe = OrdinalEncoder(categories=[['Poor', 'Average', 'Good'], ['School', 'UG', 'PG']]) # what we are doing is we are specifying the categories that we want to encode and the order of the categories, so that the encoder knows how to encode the categories in the right order, so that it can assign the right numerical values to the categories, so that it can be used in the machine learning model. Poor has the lowest value, Average has the middle value and Good has the highest value, so that it can be used in the machine learning model. School has the lowest value, UG has the middle value and PG has the highest value, so that it can be used in the machine learning model.
X_train = oe.fit_transform(X_train)
X_test = oe.transform(X_test)

print(oe.categories_)

le = LabelEncoder() # what we are doing is we are creating an instance of the LabelEncoder class, which is used to encode the target variable (y) into numerical values, so that it can be used in the machine learning model. The LabelEncoder class assigns a unique integer value to each unique category in the target variable, so that it can be used in the machine learning model.
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(le.classes_)
print(y_train)

