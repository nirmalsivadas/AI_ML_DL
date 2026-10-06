from pathlib import Path

import pandas as pd

csv_path = Path(__file__).resolve().parent / 'files' / 'matches - matches.csv'
df = pd.read_csv(csv_path)

result = (
    df.drop_duplicates(subset=['city', 'season'], keep='last')
      [['season', 'winner']]
      .sort_values(by='season')
)

print(result)