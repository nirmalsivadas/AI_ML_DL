from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

DATA_PATH = Path(__file__).resolve().parent / 'files' / 'placement (2).csv'
df = pd.read_csv(DATA_PATH)

print(df.head())
print(df.shape)

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
plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')
plt.show()

y_pred = lr.predict(X_test)
print(y_test.values)

print("MAE",mean_absolute_error(y_test,y_pred)) # Mean absolute error
print("MSE",mean_squared_error(y_test,y_pred)) # Mean squared error
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred))) # Root mean squared error
print("MSE",r2_score(y_test,y_pred)) # R2 score
r2 = r2_score(y_test,y_pred) # R2 score

# Adjusted R2 score
print(X_test.shape)

print(1 - ((1-r2)*(40-1)/(40-1-1))) # adjusted R2 score

new_df1 = df.copy()
new_df1['random_feature'] = np.random.random(200)

new_df1 = new_df1[['cgpa','random_feature','package']]
print(new_df1.head())

plt.scatter(new_df1['random_feature'],new_df1['package'])
plt.xlabel('random_feature')
plt.ylabel('Package(in lpa)')
plt.show()

X = new_df1.iloc[:,0:2]
y = new_df1.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print(1 - ((1-r2)*(40-1)/(40-1-2)))

new_df2 = df.copy()

new_df2['iq'] = new_df2['package'] + (np.random.randint(-12,12,200)/10)

new_df2 = new_df2[['cgpa','iq','package']]

print(new_df2.sample(5))

plt.scatter(new_df2['iq'],new_df2['package'])
plt.xlabel('iq')
plt.ylabel('Package(in lpa)')
plt.show()

print(np.random.randint(-100,100))

X = new_df2.iloc[:,0:2]
y = new_df2.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print("R2 score",r2_score(y_test,y_pred))
r2 = r2_score(y_test,y_pred)

print(1 - ((1-r2)*(40-1)/(40-1-2)))