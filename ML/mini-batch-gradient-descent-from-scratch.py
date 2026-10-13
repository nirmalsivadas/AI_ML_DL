from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X,y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

reg = LinearRegression()
reg.fit(X_train,y_train)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))

import random

class MBGDRegressor:
    
    def __init__(self,batch_size,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
    def fit(self,X_train,y_train):
        # init your coefs
        self.intercept_ = 0 # b0
        self.coef_ = np.ones(X_train.shape[1]) # b1,b2,b3 etc
        
        for i in range(self.epochs):
            
            for j in range(int(X_train.shape[0]/self.batch_size)): # for each data point in the batch
                
                idx = random.sample(range(X_train.shape[0]),self.batch_size) # random sample
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_ # y = b0 + b1*x1 + b2*x2
                #print("Shape of y_hat",y_hat.shape)
                intercept_der = -2 * np.mean(y_train[idx] - y_hat) # b0_der = -2 * mean(y - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der) # b0 = b0 - lr * b0_der

                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx]) # b1_der = -2 * sum(y - y_hat) * x1
                self.coef_ = self.coef_ - (self.lr * coef_der) # b1 = b1 - lr * b1_der
        
        print(self.intercept_,self.coef_)
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_

mbr = MBGDRegressor(batch_size=int(X_train.shape[0]/50),learning_rate=0.01,epochs=100)

print(mbr.fit(X_train,y_train))

y_pred = mbr.predict(X_test)
print(r2_score(y_test,y_pred))

sgd = SGDRegressor(learning_rate='constant',eta0=0.1)

batch_size = 35

for i in range(100):
    
    idx = random.sample(range(X_train.shape[0]),batch_size)
    sgd.partial_fit(X_train[idx],y_train[idx])

print(sgd.coef_)

y_pred = sgd.predict(X_test)
print(r2_score(y_test,y_pred))