from pathlib import Path

import pandas as pd

csv_path = Path(__file__).resolve().parent / 'files' / 'matches - matches.csv'
df = pd.read_csv(csv_path)

numeric_df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))

if numeric_df.select_dtypes(include='number').empty:
    raise ValueError('No numeric columns were found in the CSV file.')

print(numeric_df.corr())