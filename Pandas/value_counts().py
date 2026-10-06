import numpy as np
import pandas as pd

df = pd.read_csv('files/matches - matches.csv') # read the csv file

print(df['winner'].value_counts())