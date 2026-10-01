import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('files/cars.csv')

print(df.sample())

print(df['owner'].value_counts())

pd.get_dummies(df,columns=['fuel','owner']) # what we are doing is we are using the get_dummies function from pandas to create dummy variables for the categorical columns 'fuel' and 'owner', so that they can be used in the machine learning model. The get_dummies function creates a new column for each unique category in the categorical columns, and assigns a value of 1 or 0 to each row, depending on whether the row belongs to that category or not. This way, we can represent the categorical variables as numerical variables, which can be used in the machine learning model.

#k-1 one hot encoding
pd.get_dummies(df,columns=['fuel','owner'],drop_first=True) # what we are doing is we are using the get_dummies function from pandas to create dummy variables for the categorical columns 'fuel' and 'owner', so that they can be used in the machine learning model. The get_dummies function creates a new column for each unique category in the categorical columns, and assigns a value of 1 or 0 to each row, depending on whether the row belongs to that category or not. This way, we can represent the categorical variables as numerical variables, which can be used in the machine learning model. The drop_first parameter is set to True, which means that we are dropping the first category of each categorical column, so that we have k-1 dummy variables for each categorical column, where k is the number of unique categories in the column. This is done to avoid multicollinearity in the machine learning model.

X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,0:4],df.iloc[:,-1],test_size=0.2,random_state=2)

print(X_train.head())

ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32) # we use it so that it can store the data in a more efficient way.

X_train_new = ohe.fit_transform(X_train[['fuel','owner']])
X_test_new = ohe.transform(X_test[['fuel','owner']])

print(np.hstack((X_train[['brand','km_driven']].values,X_train_new)))

#one hot encoding with top categories
counts = df['brand'].value_counts()
nunique = df['brand'].nunique()
threshold = 100

repl = counts[counts <= threshold].index

print(pd.get_dummies(df['brand'].replace(repl, 'uncommon')).sample(5))
