from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor as SklearnSGDRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time

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

class SGDRegressorFromScratch:
    
    def __init__(self,learning_rate=0.01,epochs=100):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,X_train,y_train):
        # init your coefs
        self.intercept_ = 0 # b0
        self.coef_ = np.ones(X_train.shape[1]) # b1,b2,b3 etc
        
        for i in range(self.epochs): # for each epoch or for each iteration
            for j in range(X_train.shape[0]): # for each data point
                idx = np.random.randint(0,X_train.shape[0]) # select a random data point
                
                y_hat = np.dot(X_train[idx],self.coef_) + self.intercept_ # y = b0 + b1*x1 + b2*x2
                
                intercept_der = -2 * (y_train[idx] - y_hat) # b0_der = -2 * (y - y_hat)
                self.intercept_ = self.intercept_ - (self.lr * intercept_der) # b0 = b0 - lr * b0_der
                
                coef_der = -2 * np.dot((y_train[idx] - y_hat),X_train[idx]) # b1_der = -2 * (y - y_hat) * x1
                self.coef_ = self.coef_ - (self.lr * coef_der) # b1 = b1 - lr * b1_der
        
        print(self.intercept_,self.coef_) # print the intercept and the coefs
    
    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_ # y = b0 + b1*x1 + b2*x2

sgd = SGDRegressorFromScratch(learning_rate=0.01,epochs=40)
start = time.time() # record the start time
sgd.fit(X_train,y_train) # fit the model
print("The time taken is",time.time() - start)

y_pred = sgd.predict(X_test) # predict
print(r2_score(y_test,y_pred)) # print the r2 score

reg = SklearnSGDRegressor(max_iter=100,learning_rate='constant',eta0=0.01) # eta0 is the learning rate
print(reg.fit(X_train,y_train))

y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))