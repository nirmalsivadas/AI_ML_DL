import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('files/train.csv')

print(df.head())

sns.countplot(data=df, x='Embarked') # bar chart
plt.show()

df['Sex'].value_counts().plot(kind='pie',autopct='%.2f') # pie chart
plt.show() 

plt.hist(df['Age'],bins=5) # histogram
plt.show()

sns.histplot(df['Age'], bins=5) # distribution plot
plt.show()

sns.boxplot(data=df, x='Age') # box plot
plt.show()

print(df['Age'].min())
print(df['Age'].max())
print(df['Age'].mean())
print(df['Age'].skew())
