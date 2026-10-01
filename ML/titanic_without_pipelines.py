import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv('files/train.csv')
print(df.head())
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
print(df.head())

# Step 1 -> train/test/split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'],test_size=0.2,random_state=42)

print(X_train.head())
print(y_train.head())
print(df.isnull().sum())

# Applying imputation

si_age = SimpleImputer() # mean strategy is the default strategy for SimpleImputer, which means that it will replace the missing values in the 'Age' column with the mean value of the 'Age' column. This is a common strategy for imputing missing values in numerical columns, as it helps to preserve the overall distribution of the data and reduces the impact of outliers. However, it may not be the best strategy for all datasets, and other strategies such as median or mode may be more appropriate depending on the characteristics of the data.
si_embarked = SimpleImputer(strategy='most_frequent') # The 'most_frequent' strategy is used for imputing missing values in categorical columns, such as the 'Embarked' column in this case. It replaces the missing values with the most frequently occurring value in the column, which helps to preserve the overall distribution of the data and reduces the impact of outliers. This strategy is commonly used for categorical columns, as it helps to maintain the integrity of the data and ensures that the imputed values are representative of the overall distribution of the column. However, it may not be the best strategy for all datasets, and other strategies such as using a constant value or using a predictive model may be more appropriate depending on the characteristics of the data.

# Applying one hot encoding on Age and Embarked columns because they are categorical columns, and we want to convert them into numerical columns so that they can be used in the machine learning model. One hot encoding creates a new column for each unique value in the categorical column, and assigns a value of 1 or 0 to each row, depending on whether the row belongs to that category or not. This way, we can represent the categorical variables as numerical variables, which can be used in the machine learning model.
X_train_age = si_age.fit_transform(X_train[['Age']])
X_train_embarked = pd.DataFrame(
    si_embarked.fit_transform(X_train[['Embarked']]),
    columns=['Embarked'],
    index=X_train.index
)

X_test_age = si_age.transform(X_test[['Age']])
X_test_embarked = pd.DataFrame(
    si_embarked.transform(X_test[['Embarked']]),
    columns=['Embarked'],
    index=X_test.index
)

# one hot encoding Sex and Embarked columns because they are categorical columns, and we want to convert them into numerical columns so that they can be used in the machine learning model. One hot encoding creates a new column for each unique value in the categorical column, and assigns a value of 1 or 0 to each row, depending on whether the row belongs to that category or not. This way, we can represent the categorical variables as numerical variables, which can be used in the machine learning model.

ohe_sex = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_embarked = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked[['Embarked']])

X_test_sex = ohe_sex.transform(X_test[['Sex']])
X_test_embarked = ohe_embarked.transform(X_test_embarked[['Embarked']])

# drop Sex,Age,Embarked columns because we have already transformed them into numerical columns, and we don't need them anymore. We will concatenate the transformed columns with the rest of the columns to create the final training and testing datasets.
X_train_rem = X_train.drop(columns=['Sex','Age','Embarked'])

X_test_rem = X_test.drop(columns=['Sex','Age','Embarked'])

X_train_transformed = np.concatenate((X_train_rem,X_train_age,X_train_sex,X_train_embarked),axis=1)
X_test_transformed = np.concatenate((X_test_rem,X_test_age,X_test_sex,X_test_embarked),axis=1)

clf = DecisionTreeClassifier()
clf.fit(X_train_transformed,y_train)

y_pred = clf.predict(X_test_transformed)
print(y_pred)

print(accuracy_score(y_test,y_pred))

# Saving the models using pickle so that we can use them later for prediction on new data. We are saving the SimpleImputer, OneHotEncoder and DecisionTreeClassifier models separately, so that we can use them later for prediction on new data. We are saving the models in the 'models' folder, which we will create in the current working directory. We are using the 'wb' mode to write the models in binary format, so that we can read them later using the 'rb' mode.
model_dir = Path(__file__).resolve().parent / 'models'
model_dir.mkdir(exist_ok=True)

pickle.dump(ohe_sex, open(model_dir / 'ohe_sex.pkl', 'wb'))
pickle.dump(ohe_embarked, open(model_dir / 'ohe_embarked.pkl', 'wb'))
pickle.dump(clf, open(model_dir / 'clf.pkl', 'wb'))
