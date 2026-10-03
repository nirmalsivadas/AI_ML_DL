import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('files/placement (1).csv')

print(df.head())

sns.displot(df['cgpa'], kde=True, height=4, aspect=2)
plt.title('cgpa distribution')
plt.show()

sns.displot(df['placement_exam_marks'], kde=True, height=4, aspect=2)
plt.title('placement exam marks distribution')
plt.show()

print(df['placement_exam_marks'].describe())

sns.boxplot(x=df['placement_exam_marks'])
plt.show()

# Finding the IQR
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)

print(percentile75)

iqr = percentile75 - percentile25
print(iqr)

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print("Upper limit", upper_limit)
print("Lower limit", lower_limit)

print(df[df['placement_exam_marks'] > upper_limit])
print(df[df['placement_exam_marks'] < lower_limit])

new_df = df[df['placement_exam_marks'] < upper_limit]
print(new_df.shape)

# Comparing

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(df['placement_exam_marks'], kde=True, ax=axes[0])
axes[0].set_title('Before IQR trimming')

sns.histplot(new_df['placement_exam_marks'], kde=True, ax=axes[1])
axes[1].set_title('After IQR trimming')
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
)

print(new_df_cap.shape)

# Comparing

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(df['placement_exam_marks'], kde=True, ax=axes[0])
axes[0].set_title('Before capping')

sns.histplot(new_df_cap['placement_exam_marks'], kde=True, ax=axes[1])
axes[1].set_title('After capping')
plt.show()

