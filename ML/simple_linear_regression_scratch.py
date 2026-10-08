class MeraLR:
    
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self,X_train,y_train):
        
        num = 0
        den = 0
        
        for i in range(X_train.shape[0]):
            
            num = num + ((X_train[i] - X_train.mean())*(y_train[i] - y_train.mean())) # this is the numerator
            den = den + ((X_train[i] - X_train.mean())*(X_train[i] - X_train.mean())) # this is the denominator
        
        self.m = num/den # this is the slope
        self.b = y_train.mean() - (self.m * X_train.mean()) # this is the intercept
        print(self.m)
        print(self.b)       
    
    def predict(self,X_test):
        
        print(X_test)
        
        return self.m * X_test + self.b
  
import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path(__file__).resolve().parent / 'files' / 'placement (2).csv'
df = pd.read_csv(csv_path)
X = df.iloc[:,0].values
y = df.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
lr = MeraLR()
lr.fit(X_train,y_train)
print(lr.predict(X_test[0]))