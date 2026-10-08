import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('files/placement (2).csv')
print(df.head())

plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()

X = df.iloc[:,0:1]
y = df.iloc[:,-1]

print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
print(X_test)

print(y_test)

print(lr.predict(X_test.iloc[0].values.reshape(1,1))) # lr.predict is a function that returns the predicted values of y for the given values of x

plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()

m = lr.coef_
b = lr.intercept_

# y = mx + b

print(m * 8.58 + b)

print(m * 9.5 + b)
print(m * 100 + b)

