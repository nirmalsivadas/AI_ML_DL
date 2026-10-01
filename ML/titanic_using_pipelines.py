import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import set_config
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('files/train.csv')
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']), df['Survived'],test_size=0.2,random_state=42)

# imputation transformer
trf1 = ColumnTransformer([
    ('impute_age',SimpleImputer(),[2]), # impute_age is the name of the transformer, SimpleImputer() is the transformer object, and [2] is the list of columns to be transformed. The SimpleImputer() object is used to impute missing values in the 'Age' column, which is the 3rd column in the dataframe (index 2). The default strategy for SimpleImputer() is 'mean', which means that it will replace the missing values in the 'Age' column with the mean value of the 'Age' column. This is a common strategy for imputing missing values in numerical columns, as it helps to preserve the overall distribution of the data and reduces the impact of outliers. However, it may not be the best strategy for all datasets, and other strategies such as median or mode may be more appropriate depending on the characteristics of the data.
    ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])
],remainder='passthrough') # what we are doing is we are creating a column transformer object, which will apply the specified transformations to the specified columns of the dataframe, and will leave the rest of the columns unchanged. The remainder parameter is set to 'passthrough', which means that the rest of the columns will be left unchanged. This is better than the previous approach, because we don't have to manually concatenate the transformed columns with the rest of the columns, and we can also easily change the transformations applied to the columns, without having to change the code for concatenating the columns. this is our first step in the pipeline, which is to impute the missing values in the 'Age' and 'Embarked' columns, so that we can use them in the machine learning model. The SimpleImputer class is used to impute the missing values, and we are using the default strategy of 'mean' for the 'Age' column, and the 'most_frequent' strategy for the 'Embarked' column. This means that we will replace the missing values in the 'Age' column with the mean value of the column, and we will replace the missing values in the 'Embarked' column with the most frequently occurring value in the column. This is a common strategy for imputing missing values in numerical and categorical columns, as it helps to preserve the overall distribution of the data and reduces the impact of outliers. However, it may not be the best strategy for all datasets, and other strategies such as median or mode may be more appropriate depending on the characteristics of the data.

# one hot encoding
trf2 = ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [1, 6])
], remainder='passthrough') # what we are doing is we are creating a column transformer object, which will apply the specified transformations to the specified columns of the dataframe, and will leave the rest of the columns unchanged. The remainder parameter is set to 'passthrough', which means that the rest of the columns will be left unchanged. This is better than the previous approach, because we don't have to manually concatenate the transformed columns with the rest of the columns, and we can also easily change the transformations applied to the columns, without having to change the code for concatenating the columns. this is our second step in the pipeline, which is to one hot encode the 'Sex' and 'Embarked' columns, so that we can use them in the machine learning model. The OneHotEncoder class is used to one hot encode the categorical columns, and we are using the handle_unknown='ignore' parameter to ignore any unknown categories that may be present in the test data. This means that if there are any categories in the test data that were not present in the training data, they will be ignored and not included in the one hot encoded representation. This is a common strategy for handling unknown categories in categorical columns, as it helps to prevent errors when making predictions on new data. However, it may not be the best strategy for all datasets, and other strategies such as using a constant value or using a predictive model may be more appropriate depending on the characteristics of the data.

# Scaling
trf3 = ColumnTransformer([
    ('scale',MinMaxScaler(),slice(0,10))
]) 

# Feature selection
trf4 = SelectKBest(score_func=chi2,k=8) # select the top 8 features, what this does is it selects the top 8 features based on the chi-squared statistic, which measures the dependence between the features and the target variable. The chi-squared statistic is a measure of how much the observed frequencies of the features differ from the expected frequencies, and it is used to determine which features are most relevant for predicting the target variable. The SelectKBest class is used to select the top k features based on the chi-squared statistic, and we are setting k=8 to select the top 8 features. This is a common strategy for feature selection in machine learning, as it helps to reduce the dimensionality of the data and improve the performance of the model. However, it may not be the best strategy for all datasets, and other strategies such as recursive feature elimination or feature importance may be more appropriate depending on the characteristics of the data.

# train the model
trf5 = DecisionTreeClassifier() # this is our final step in the pipeline, which is to train the decision tree classifier

pipe = Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])

# # Alternate Syntax, Pipeline requires naming of steps, make_pipeline does not.
#(Same applies to ColumnTransformer vs make_column_transformer)
# pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5)

# train
pipe.fit(X_train,y_train)

# Code here
print(pipe.named_steps) # pipe.named_steps

# Display Pipeline
set_config(display='diagram')

# Predict
y_pred = pipe.predict(X_test)

print(y_pred)

print(accuracy_score(y_test,y_pred))

# cross validation using cross_val_score
print(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())

# gridsearchcv
params = {
    'trf5__max_depth':[1,2,3,4,5,None]
}

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

# export 
pickle.dump(pipe,open('models/pipe.pkl','wb'))
