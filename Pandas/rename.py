import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

print(df.rename(columns={'team1': 'team_a', 'team2': 'team_b'}))