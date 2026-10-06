import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file
print(df.shape)
df.dropna(subset=['team1', 'team2'])
print(df.shape)

df.fillna(0) # will replace all NaN values with 0
print(df.shape)

df.fillna(subset=['team1', 'team2'], method='ffill')
print(df.shape)

df['team1'].fillna('unknown')
print(df.shape)