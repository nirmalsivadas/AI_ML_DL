
import pandas as pd
from pandas_datareader import data
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV

boston = load_boston()
df = pd.DataFrame(boston.data)
     

df.columns = boston.feature_names
df['MEDV'] = boston.target
     

df.head()

X = df.iloc[:,0:13]
y = df.iloc[:,13]
     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
     

rt = DecisionTreeRegressor(criterion = 'mse', max_depth=5)
     

rt.fit(X_train,y_train)

y_pred = rt.predict(X_test)
     

r2_score(y_test,y_pred)

param_grid = {
    'max_depth':[2,4,8,10,None],
    'criterion':['mse','mae'],
    'max_features':[0.25,0.5,1.0],
    'min_samples_split':[0.25,0.5,1.0]
}
     

reg = GridSearchCV(DecisionTreeRegressor(),param_grid=param_grid)
     

reg.fit(X_train,y_train)

reg.best_score_
     
reg.best_params_

for importance, name in sorted(zip(rt.feature_importances_, X_train.columns),reverse=True):
  print (name, importance)