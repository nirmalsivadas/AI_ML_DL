from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer

base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / 'files' / 'train.csv', usecols=['Age', 'Fare', 'Survived'])

df.dropna(inplace=True) # drop rows with missing values
print(df.shape)
print(df.head())

X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.head(2))

clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,y_pred))

print('Cross-Validation Accuracy:',np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy'))) # what this does is it performs cross validation on the entire dataset X and y using a DecisionTreeClassifier, with 10 folds (cv=10), and calculates the accuracy score for each fold. It then returns the mean accuracy across all folds, providing an estimate of the model's performance on unseen data.

kbin_age = KBinsDiscretizer(n_bins=15,encode='ordinal',strategy='quantile') 
kbin_fare = KBinsDiscretizer(n_bins=15,encode='ordinal',strategy='quantile') # what both of these lines do is they create two instances of the KBinsDiscretizer class from scikit-learn. The first instance, kbin_age, is configured to discretize the 'Age' feature into 15 bins using the 'quantile' strategy and encode the bins as ordinal values. The second instance, kbin_fare, does the same for the 'Fare' feature. This means that both features will be transformed into discrete ordinal values based on their distribution in the dataset.

trf = ColumnTransformer([
    ('first',kbin_age,[0]),
    ('second',kbin_fare,[1])
]) # what this line does is it creates an instance of the ColumnTransformer class from scikit-learn. The ColumnTransformer allows you to apply different transformations to different columns of your dataset. In this case, it applies the kbin_age transformation to the first column (index 0, which corresponds to 'Age') and the kbin_fare transformation to the second column (index 1, which corresponds to 'Fare'). The resulting transformed data will have the same number of rows as the original data but with the specified columns transformed into discrete ordinal values based on their respective binning strategies.

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.transform(X_test)

print(trf.named_transformers_['first'].bin_edges_)
print(trf.named_transformers_['second'].bin_edges_)

output = pd.DataFrame({
    'age':X_train['Age'],
    'age_trf':X_train_trf[:,0],
    'fare':X_train['Fare'],
    'fare_trf':X_train_trf[:,1]
}) # what this line does is it creates a new pandas DataFrame called output. This DataFrame contains four columns: 'age', which holds the original 'Age' values from the X_train dataset; 'age_trf', which holds the transformed 'Age' values after applying the KBinsDiscretizer; 'fare', which holds the original 'Fare' values from the X_train dataset; and 'fare_trf', which holds the transformed 'Fare' values after applying the KBinsDiscretizer. The transformed values are obtained from the first and second columns of the X_train_trf array, respectively. This allows for a comparison between the original and transformed values for both features.

output['age_labels'] = pd.cut(x=X_train['Age'],
                                    bins=trf.named_transformers_['first'].bin_edges_[0].tolist())
output['fare_labels'] = pd.cut(x=X_train['Fare'],
                                    bins=trf.named_transformers_['second'].bin_edges_[0].tolist()) # what these two lines do is they create two new columns in the output DataFrame: 'age_labels' and 'fare_labels'. The pd.cut() function is used to bin the original 'Age' and 'Fare' values from the X_train dataset into discrete intervals based on the bin edges obtained from the KBinsDiscretizer transformations. The bin edges are accessed from the named_transformers_ attribute of the ColumnTransformer instance (trf). The resulting 'age_labels' and 'fare_labels' columns contain categorical labels representing the bins into which each original value falls, allowing for a clear understanding of how the continuous features have been discretized.

print(output.sample(5))

clf = DecisionTreeClassifier()
clf.fit(X_train_trf,y_train)
y_pred2 = clf.predict(X_test_trf)

print('Accuracy after binning:',accuracy_score(y_test,y_pred2))

X_trf = trf.fit_transform(X) # what this line does is it applies the ColumnTransformer transformation to the entire dataset X, which contains both 'Age' and 'Fare' features. The resulting transformed data is stored in X_trf.
print('Cross-Validation Accuracy after binning:',np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy')))

def discretize(bins,strategy):
    kbin_age = KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy=strategy)
    kbin_fare = KBinsDiscretizer(n_bins=bins,encode='ordinal',strategy=strategy)
    
    trf = ColumnTransformer([
        ('first',kbin_age,[0]),
        ('second',kbin_fare,[1])
    ])
    
    X_trf = trf.fit_transform(X)
    print(np.mean(cross_val_score(DecisionTreeClassifier(),X,y,cv=10,scoring='accuracy')))
    
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.hist(X['Age'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,0],color='red')
    plt.title("After")

    plt.show()
    
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.hist(X['Fare'])
    plt.title("Before")

    plt.subplot(122)
    plt.hist(X_trf[:,1],color='red')
    plt.title("Fare")
    plt.show()
    # what this function does is it takes two parameters: bins and strategy. It creates two instances of the KBinsDiscretizer class for the 'Age' and 'Fare' features, using the specified number of bins and strategy. It then creates a ColumnTransformer to apply these transformations to the respective columns. The transformed data is obtained by fitting and transforming the entire dataset X. The function prints the mean cross-validation accuracy of a DecisionTreeClassifier on the original dataset X and y. It also generates histograms to visualize the distribution of the original and transformed 'Age' and 'Fare' features before and after discretization.

print(discretize(5,'kmeans'))



