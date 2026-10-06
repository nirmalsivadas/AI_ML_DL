import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('files/matches - matches.csv') # read the csv file

df['winner'].value_counts().plot(kind='bar')
plt.show()

df['toss_decision'].value_counts().plot(kind='pie')
plt.show()

df['win_by_runs'].plot(kind='hist')
plt.show()