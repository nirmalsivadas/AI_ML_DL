import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

def get_city(city):
  mask = df['city']==city
  return df[mask]

print(get_city('Mumbai'))

mask1 = df['city']=='Hyderabad'
mask2 = df['date']>='2022-01-01'

print(df[mask1 & mask2])