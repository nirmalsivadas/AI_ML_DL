from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / 'files' / 'deliveries.csv'

df = pd.read_csv(csv_path)

death_over = df[df['over'] > 15]
all_batsman = death_over.groupby('batsman')['batsman_runs'].count()
print(all_batsman)

x = all_batsman > 200
batsman_list = all_batsman[x].index.to_list()

final = df[df['batsman'].isin(batsman_list)]
print(final)

runs = final.groupby('batsman')['batsman_runs'].sum()
balls = final.groupby('batsman')['batsman_runs'].count()
sr = (runs / balls) * 100
print(sr)

