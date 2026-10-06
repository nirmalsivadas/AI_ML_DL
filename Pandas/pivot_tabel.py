import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

print(df.pivot_table(index='team1', columns='team2', values='winner', aggfunc='count'))

print(df.pivot_table(index=['city', 'season'], columns=['team1', 'team2'], values='winner', aggfunc='count'))