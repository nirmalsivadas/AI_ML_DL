
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('files/placement (1).csv')

print(df.head())

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])

plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])

plt.show()

plot = sns.displot(df['placement_exam_marks'], kde=True, height=4, aspect=2)
plt.title('placement exam marks distribution')
plt.show()

print(df['placement_exam_marks'].describe())

sns.boxplot(df['placement_exam_marks'])
plt.show()

# Finding the IQR
percentile25 = df['placement_exam_marks'].quantile(0.25) # 25th percentile 
percentile75 = df['placement_exam_marks'].quantile(0.75) # 75th percentile

print(percentile75)

iqr = percentile75 - percentile25
print(iqr)

upper_limit = percentile75 + 1.5 * iqr # any value greater than upper limit is outlier
lower_limit = percentile25 - 1.5 * iqr # any value less than lower limit is outlier

print("Upper limit", upper_limit)
print("Lower limit", lower_limit)

print(df[df['placement_exam_marks'] > upper_limit])
print(df[df['placement_exam_marks'] < lower_limit])

new_df = df[df['placement_exam_marks'] < upper_limit] # removing the outliers
print(new_df.shape)

# Comparing

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df['placement_exam_marks'])

plt.show()

new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
) # capping the outliers

print(new_df_cap.shape)

# Comparing

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot(df['placement_exam_marks'])

plt.subplot(2,2,2)
sns.boxplot(df['placement_exam_marks'])

plt.subplot(2,2,3)
sns.distplot(new_df_cap['placement_exam_marks'])

plt.subplot(2,2,4)
sns.boxplot(new_df_cap['placement_exam_marks'])

plt.show()