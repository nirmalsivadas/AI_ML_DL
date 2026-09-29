import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips') # load dataset basically loads specific dataset from seaborn

print(tips.head())

titanic = pd.read_csv("C:/Users/Nirmal/OneDrive/Desktop/AI_ML_DL/ML/files/train.csv")

flights = sns.load_dataset('flights')

print(flights.head())

iris = sns.load_dataset('iris')

print(iris.head())

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', style='smoker', size='size') # scatter plot

plt.show()

print(titanic.head())

sns.barplot(data=titanic, x='Pclass', y='Age', hue='Sex') # bar plot

plt.show()


sns.boxplot(data=titanic, x='Sex', y='Age', hue='Survived')

plt.show()

sns.kdeplot(titanic[titanic['Survived'] == 0]['Age'], label='Survived=0', warn_singular=False)
sns.kdeplot(titanic[titanic['Survived'] == 1]['Age'], label='Survived=1', warn_singular=False)
plt.show()

print(titanic.head(3))
sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))

plt.show()

print((titanic.groupby('Embarked')['Survived'].mean() * 100))


sns.clustermap(pd.crosstab(titanic['Parch'],titanic['Survived']))

plt.show()

sns.pairplot(iris,hue='species')

plt.show()

new = flights.groupby('year', as_index=False)['passengers'].sum()

sns.lineplot(data=new, x='year', y='passengers')

plt.show()


sns.clustermap(flights.pivot_table(values='passengers',index='month',columns='year'))

plt.show()




