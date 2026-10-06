import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

print(df['team1'].value_counts() + df['team2'].value_counts())