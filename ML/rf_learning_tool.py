import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
     

X,y = make_classification(n_features=5, n_redundant=0, n_informative=5,n_clusters_per_class=1)
     

df = pd.DataFrame(X,columns=['col1','col2','col3','col4','col5'])
df['target'] = y
print(df.shape)
print(df.head())

# function for row sampling

def sample_rows(df,percent):
  return df.sample(int(percent*df.shape[0]),replace=True)
     

# function for feature sampling
def sample_features(df,percent):
  cols = random.sample(df.columns.tolist()[:-1],int(percent*(df.shape[1]-1)))
  new_df = df[cols].copy()
  new_df['target'] = df['target']
  return new_df
     

# function for combined sampling

def combined_sampling(df,row_percent,col_percent):
  new_df = sample_rows(df,row_percent)
  return sample_features(new_df,col_percent)
     

df1 = combined_sampling(df,0.5,0.5)

df2 = combined_sampling(df,0.5,0.5)
df3 = combined_sampling(df,0.5,0.5)


print(df1.columns)
print(df2.columns)
print(df3.columns)

print(df3)

from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()
clf3 = DecisionTreeClassifier()
     

clf1.fit(df1.iloc[:,0:2],df1.iloc[:,-1])
clf2.fit(df2.iloc[:,0:2],df2.iloc[:,-1])
clf3.fit(df3.iloc[:,0:2],df3.iloc[:,-1])

from sklearn.tree import plot_tree
     

plot_tree(clf1)
plt.show()

plot_tree(clf2)
plt.show()

plot_tree(clf3)
plt.show()

sample = [-1.042498, -0.064323]
clf1.predict(pd.DataFrame([sample], columns=df1.columns[:2]))
clf2.predict(pd.DataFrame([sample], columns=df2.columns[:2]))
clf3.predict(pd.DataFrame([sample], columns=df3.columns[:2]))

print(df.sample(14,replace=True))