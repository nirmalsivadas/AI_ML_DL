import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

print(df.sort_values('city', ascending=False))
print(df.sort_values(['city','date'], ascending=[True,False]))