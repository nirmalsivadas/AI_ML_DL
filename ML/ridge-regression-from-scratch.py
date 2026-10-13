from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np

X,y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4) # splitting the data

from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1,solver='cholesky') # solver = 'cholesky' means using Cholesky decomposition to solve the equation. lambda = alpha
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))

print(reg.coef_)
print(reg.intercept_)

class MeraRidge:
    
    def __init__(self,alpha=0.1):
        
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self,X_train,y_train):
        
        X_train = np.insert(X_train,0,1,axis=1) # adding a column of 1s
        I = np.identity(X_train.shape[1]) # identity matrix
        I[0][0] = 0
        result = np.linalg.inv(np.dot(X_train.T,X_train) + self.alpha * I).dot(X_train.T).dot(y_train) # w = (X^T * X + alpha * I)^-1 * X^T * y
        self.intercept_ = result[0] # b0
        self.coef_ = result[1:] # b1,b2,b3
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_

reg = MeraRidge()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))
print(reg.coef_)
print(reg.intercept_)