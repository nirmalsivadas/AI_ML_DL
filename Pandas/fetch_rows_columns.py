import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

data1 = df['winner'] # fetch one column
print(data1)

data2 = df[['team1','team2','winner']] # fetch multiple columns
print(data2)

data3 = df.iloc[0] # fetch one row
print(data3)

data4 = df.iloc[0:2] # fetch multiple rows
print(data4)

data5 = df.iloc[0:6:2] # fetch multiple rows with step size 2
print(data5)

data6 = df.iloc[[1,5,6]] # fetch multiple rows by index
print(data6)

data7 = df.iloc[:,[4,5,10]] # fetch multiple columns by index and rows
print(data7)