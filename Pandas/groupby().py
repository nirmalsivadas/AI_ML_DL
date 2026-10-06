from pathlib import Path

import numpy as np
import pandas as pd

base_dir = Path(__file__).resolve().parent

# Fortune 500 dataset
fortune_path = base_dir / 'files' / 'Fortune 500 Companies US.csv'
df = pd.read_csv(fortune_path, encoding='cp1252')

# Clean numeric-looking columns for aggregation
for col in ['Number of Employees', 'Revenues', 'Profits', 'Assets', 'Market Value']:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('[,$%\s]', '', regex=True)
            .apply(pd.to_numeric, errors='coerce')
        )

# Use a valid grouping column present in the dataset
group_col = 'Previous Rank' if 'Previous Rank' in df.columns else df.columns[0]
fortune_grouped = df.groupby(group_col)

print('FORTUNE GROUPBY OBJECT:')
print(fortune_grouped)
print('Row count:', len(df))
print('Group count:', len(fortune_grouped))
print('Top groups by size:')
print(fortune_grouped.size().sort_values(ascending=False).head())
print('First rows per group:')
print(fortune_grouped.first().head())
print('Last rows per group:')
print(fortune_grouped.last().head())
print('Example group:')
print(fortune_grouped.get_group(1.0).head())
print('Total revenues by previous rank:')
print(fortune_grouped['Revenues'].sum().sort_values(ascending=False).head())
print('Mean of numeric columns:')
print(df.select_dtypes(include=[np.number]).mean(numeric_only=True).head())

# IPL deliveries dataset
ipl_path = base_dir / 'files' / 'deliveries.csv'
df1 = pd.read_csv(ipl_path)
runs = df1.groupby('batsman')
print('\nTOP RUN SCORERS:')
print(runs['batsman_runs'].sum().sort_values(ascending=False).head(5))

mask = df1['batsman_runs'] == 4
new_df = df1[mask]
print('Fours count:', new_df.shape[0])
print('Top batsmen with 4s:')
print(new_df.groupby('batsman')['batsman_runs'].count().sort_values(ascending=False).head(5))

batsman_mask = df1['batsman'] == 'V Kohli'
new_df1 = df1[batsman_mask]
print('V Kohli team run totals:')
print(new_df1.groupby('batting_team')['batsman_runs'].sum().sort_values(ascending=False).head(5))

