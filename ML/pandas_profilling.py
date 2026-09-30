import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('files/train.csv')
print(df.head())


prof = ProfileReport(df)
prof.to_file(output_file='files/output.html')