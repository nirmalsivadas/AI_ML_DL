from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / 'files' / 'titanic.csv')
print(df['number'].unique()) # what this line does is it reads a CSV file named 'titanic.csv' located in the 'files' directory into a pandas DataFrame called df. Then, it retrieves the unique values from the 'number' column of the DataFrame and returns them as a numpy array. This allows you to see all the distinct values present in that specific column.

fig = df['number'].value_counts().plot.bar()
fig.set_title('Passengers travelling with')
plt.show() # what this line does is it creates a bar plot of the value counts of the 'number' column in the DataFrame df. The plot displays the frequency of each unique value in the 'number' column, and the title of the plot is set to 'Passengers travelling with'. Finally, plt.show() is called to display the plot.

# extract numerical part
df['number_numerical'] = pd.to_numeric(df["number"],errors='coerce',downcast='integer') # what this line does is it creates a new column in the DataFrame df called 'number_numerical'. It converts the values in the 'number' column to numeric data types (specifically integers) using the pd.to_numeric() function. The errors='coerce' parameter ensures that any non-numeric values are replaced with NaN (Not a Number), and the downcast='integer' parameter attempts to downcast the resulting numeric values to the smallest integer subtype possible. This allows for easier numerical analysis of the 'number' column.

# extract categorical part
df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'],np.nan) # what this line does is it creates a new column in the DataFrame df called 'number_categorical'. It uses the np.where() function to check if the values in the 'number_numerical' column are NaN (which indicates that the original 'number' value was non-numeric). If a value is NaN, it assigns the corresponding value from the 'number' column to 'number_categorical'; otherwise, it assigns NaN. This effectively separates the categorical part of the 'number' column from the numerical part, allowing for easier analysis of categorical data.

print(df['Cabin'].unique())

print(df['Ticket'].unique())

df['cabin_num'] = df['Cabin'].str.extract(r'(\d+)') # captures numerical part
df['cabin_cat'] = df['Cabin'].str[0] # captures the first letter # what both of these lines do is they create two new columns in the DataFrame df called 'cabin_num' and 'cabin_cat'. The 'cabin_num' column extracts the numerical part of the 'Cabin' column using the str.extract() method with a regular expression that captures one or more digits (\d+). The 'cabin_cat' column captures the first letter of the 'Cabin' column using str[0], which accesses the first character of each string in the 'Cabin' column. This allows for separate analysis of the numerical and categorical components of the 'Cabin' data.

df['cabin_cat'].value_counts().plot(kind='bar')
plt.show()

# extract the last bit of ticket as number
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],
                                   errors='coerce',
                                   downcast='integer')
# what both of these lines do is they create a new column in the DataFrame df called 'ticket_num'. The first line uses the apply() method with a lambda function to split each value in the 'Ticket' column by spaces and extract the last part ([-1]), which is assumed to be the numerical part of the ticket. The second line converts the extracted values in 'ticket_num' to numeric data types (specifically integers) using pd.to_numeric(), with errors='coerce' to replace any non-numeric values with NaN and downcast='integer' to attempt to downcast the resulting numeric values to the smallest integer subtype possible. This allows for easier numerical analysis of the ticket numbers.

# extract the first part of ticket as category
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(), np.nan,
                              df['ticket_cat'])
# what both of these lines do is they create a new column in the DataFrame df called 'ticket_cat'. The first line uses the apply() method with a lambda function to split each value in the 'Ticket' column by spaces and extract the first part ([0]), which is assumed to be the categorical part of the ticket. The second line uses np.where() to check if the extracted value in 'ticket_cat' is a digit (using str.isdigit()). If it is a digit, it assigns NaN to that entry; otherwise, it keeps the extracted value. This effectively separates the categorical part of the ticket from any numerical part, allowing for easier analysis of categorical ticket data.

print(df.head(20))

print(df['ticket_cat'].unique())


