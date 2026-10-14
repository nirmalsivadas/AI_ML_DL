from pathlib import Path

import numpy as np
import pandas as pd

csv_path = Path(__file__).resolve().parent / 'files' / 'play_tennis.csv'
df = pd.read_csv(csv_path)

print(df.head())

df.drop(columns=['day'],inplace=True)

print(df['play'].value_counts())

py = 9/14
pn = 5/14

print(py,pn)

outlook = pd.crosstab(df['outlook'],df['play'])
print(outlook)

pon = 0
prn = 2/5
psn = 3/5

poy = 4/9
pry = 3/9
psy = 2/9

temp = pd.crosstab(df['temp'],df['play'])
print(temp)

pCoolNo = 1/5
pMildNo = 2/5
pHotNo = 2/5

pCoolYes = 3/9
pMildYes = 4/9
pHotYes = 2/9

humidity = pd.crosstab(df['humidity'],df['play'])
print(humidity)

pHighNo = 4/5
pNormalNo = 1/5

pHighYes = 3/9
pNormalYes = 6/9

wind = pd.crosstab(df['wind'],df['play'])
print(wind)

pWeakNo = 3/5
pStrongNo = 2/5

pWeakYes = 3/9
pStrongYes = 6/9

pyes = py*psy*pHotYes,pHighYes*pWeakYes
print(pyes)
pno = pn*psn*pHotNo,pHighNo*pWeakNo
print(pno)