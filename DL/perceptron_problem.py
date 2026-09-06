import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

or_data = pd.DataFrame()
and_data = pd.DataFrame()
xor_data = pd.DataFrame()

or_data['Feature1'] = [0, 0, 1, 1]
or_data['Feature2'] = [0, 1, 0, 1]
or_data['output'] = [0, 1, 1, 1]

and_data['Feature1'] = [0, 0, 1, 1]
and_data['Feature2'] = [0, 1, 0, 1]
and_data['output'] = [0, 0, 0, 1]

xor_data['Feature1'] = [0, 0, 1, 1]
xor_data['Feature2'] = [0, 1, 0, 1]
xor_data['output'] = [0, 1, 1, 0]

print("OR Data:")
print(or_data)

sns.scatterplot(data=or_data, x=or_data['Feature1'], y=or_data['Feature2'], hue=or_data['output'], s=200)
plt.title("OR Gate Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("AND Data:")
print(and_data)

sns.scatterplot(data=and_data, x=and_data['Feature1'], y=and_data['Feature2'], hue=and_data['output'], s=200)
plt.title("AND Gate Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("XOR Data:")
print(xor_data)

sns.scatterplot(data=xor_data, x=xor_data['Feature1'], y=xor_data['Feature2'], hue=xor_data['output'], s=200)
plt.title("XOR Gate Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

clf1 = Perceptron(random_state=0)
clf2 = Perceptron(random_state=0)
clf3 = Perceptron(random_state=0)

# Keep the notebook's model order: clf1 = AND, clf2 = OR, clf3 = XOR.
clf1.fit(and_data.iloc[:,0:2].values, and_data.iloc[:,-1].values)
clf2.fit(or_data.iloc[:,0:2].values, or_data.iloc[:,-1].values)
clf3.fit(xor_data.iloc[:,0:2].values, xor_data.iloc[:,-1].values)

print(clf1.coef_, clf1.intercept_) #clf1.coef_ and clf1.intercept_ are the weights and bias of the perceptron model trained on the AND gate data
print(clf2.coef_, clf2.intercept_) #clf2.coef_ and clf2.intercept_ are the weights and bias of the perceptron model trained on the OR gate data
print(clf3.coef_, clf3.intercept_) #clf3.coef_ and clf3.intercept_ are the weights and bias of the perceptron model trained on the XOR gate data
print("AND accuracy:", clf1.score(and_data.iloc[:, 0:2], and_data.iloc[:, -1]))
print("OR accuracy:", clf2.score(or_data.iloc[:, 0:2], or_data.iloc[:, -1]))
print("XOR accuracy:", clf3.score(xor_data.iloc[:, 0:2], xor_data.iloc[:, -1]))

x = np.linspace(-1,1,5)
y = -x+0.5

plt.plot(x,y)
sns.scatterplot(data=or_data, x=or_data['Feature1'], y=or_data['Feature2'], hue=or_data['output'], s=200)
plt.show()

x1 = np.linspace(-1,1,5)
y1 = -x1+1.5

plt.plot(x1,y1)
sns.scatterplot(data=and_data, x=and_data['Feature1'], y=and_data['Feature2'], hue=and_data['output'], s=200)
plt.show()

x2 = np.linspace(-1,1,5)

plot_decision_regions(xor_data.iloc[:,0:2].values,xor_data.iloc[:,-1].values, clf=clf3, legend=2)
plt.title("XOR Gate Decision Regions")
plt.show()