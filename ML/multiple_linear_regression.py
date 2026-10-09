from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = make_regression(n_samples=100, n_features=2, n_informative=2, n_targets=1, noise=50) # this will generate 2 features and 1 target variable
df = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'target': y})

print(df.shape)
print(df.head())

fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')
fig.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("MAE", mean_absolute_error(y_test, y_pred))
print("MSE", mean_squared_error(y_test, y_pred))
print("R2 score", r2_score(y_test, y_pred))

x = np.linspace(-5, 5, 10) # we do this to get 10 values between -5 and 5
y_plot = np.linspace(-5, 5, 10) # we do this to get 10 values between -5 and 5
xGrid, yGrid = np.meshgrid(x, y_plot) # this will give us 10x10 grid
final = np.column_stack((xGrid.ravel(), yGrid.ravel())) # what this does is that it stacks the 10x10 grid into a 100x2 matrix
z_final = lr.predict(final).reshape(xGrid.shape) # we reshape it to 10x10
fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')
fig.add_trace(go.Surface(x=x, y=y_plot, z=z_final)) # we add the surface to the plot 
fig.show()

print(lr.coef_)
print(lr.intercept_)