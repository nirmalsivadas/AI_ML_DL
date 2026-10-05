import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('files/weight-height.csv')

print(df.head())

print(df.shape)

print(df['Height'].describe())

sns.distplot(df['Height'])
plt.show()

sns.boxplot(df['Height'])
plt.show()

upper_limit = df['Height'].quantile(0.99) # 99th percentile, highest value in the dataset, any value above 99th percentile is an outlier
print(upper_limit) 

lower_limit = df['Height'].quantile(0.01) # 1st percentile, lowest value in the dataset, any value below 1st percentile is an outlier
print(lower_limit)

new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)] # removing the outliers

print(new_df['Height'].describe())

print(df['Height'].describe())

sns.distplot(new_df['Height'])
plt.show()

sns.boxplot(new_df['Height'])
plt.show()

# Capping --> Winsorization
df['Height'] = np.where(df['Height'] >= upper_limit,
        upper_limit,
        np.where(df['Height'] <= lower_limit,
        lower_limit,
        df['Height']))

print(df.shape)

print(df['Height'].describe())

sns.distplot(df['Height'])
plt.show()

sns.boxplot(df['Height'])
plt.show()