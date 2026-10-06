from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent

df1 = pd.read_csv(base_dir / 'files' / 'matches - matches.csv')
df2 = pd.read_csv(base_dir / 'files' / 'deliveries.csv')

new_df = df1.merge(df2, left_on='id', right_on='match_id')

result = (
    new_df.groupby(['season', 'batsman'])['batsman_runs']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    .drop_duplicates(subset='season', keep='first')
    .sort_values('season')[['season', 'batsman']]
)

print(result)