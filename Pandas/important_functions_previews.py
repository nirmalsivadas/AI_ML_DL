import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

print(df.head()) # first 5 rows
print(df.tail()) # last 5 rows
print(df.head(2))
print(df.tail(2))

print(df.shape) # number of rows and columns

print(df.info()) # information about the dataframe including column names, data types, and number of non-null entries
print(df.describe()) # summary statistics for numeric columns including count, mean, std, min, and max