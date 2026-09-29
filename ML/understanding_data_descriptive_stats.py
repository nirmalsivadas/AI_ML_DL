
import pandas as pd

df = pd.read_csv('C:\\Users\\Nirmal\\OneDrive\\Desktop\\AI_ML_DL\\ML\\files\\placement.csv')

print(df.shape) # shape of the dataframe or how many rows and columns are in the dataframe

print(df.head()) # first 5 rows of the dataframe

print(df.tail()) # last 5 rows of the dataframe

print(df.sample) # random sample of the dataframe or how does the dataframe look like

print(df.info()) # information about the dataframe, including the number of non-null entries and data types for each column

print(df.isnull().sum()) # number of missing values in each column

print(df.describe()) # descriptive statistics of the dataframe, including count, mean, std, min, and max

print(df.corr()) # correlation matrix of the dataframe or how correlated the columns are with each other

print(df.duplicated().sum()) # number of duplicate rows in the dataframe

