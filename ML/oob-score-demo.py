from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / 'files' / 'heart.csv'
df = pd.read_csv(csv_path)
print(df.head())

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train,y_train)

print(rf.oob_score_)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test,y_pred))