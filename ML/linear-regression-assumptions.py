from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = Path(__file__).resolve().parent / 'files' / 'data.csv'
df = pd.read_csv(csv_path)

print(df.head())

X = df.iloc[:,0:3].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train,y_train)

# Residual
y_pred = model.predict(X_test)
residual = y_test - y_pred

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 2.5))

ax1.scatter(df['feature1'], df['target'])
ax1.set_title("Feature1")
ax2.scatter(df['feature2'], df['target'])
ax2.set_title("Feature2")
ax3.scatter(df['feature3'], df['target'])
ax3.set_title("Feature3")

plt.show() # linear relationship

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = []

for i in range(X_train.shape[1]):
    vif.append(variance_inflation_factor(X_train, i))
pd.DataFrame({'vif': vif}, index=df.columns[0:3]).T # checks for multicollinearity

# Another Technique
sns.heatmap(df.iloc[:,0:3].corr(),annot=True)
plt.show()

sns.displot(residual,kind='kde')
plt.show() # normal distribution

# QQ Plot

import scipy as sp

fig, ax = plt.subplots(figsize=(6,4))
sp.stats.probplot(residual, plot=ax, fit=True) # checks for normality

plt.show()

plt.scatter(y_pred,residual) # homoscedasticity
plt.show()

plt.plot(residual) # no autocorrelation in the residuals
plt.show()

