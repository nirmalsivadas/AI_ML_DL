from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / 'files' / 'train.csv'
df = pd.read_csv(csv_path)

print(df.head())

# Use only numeric feature columns and keep the target column as the label.
feature_cols = df.select_dtypes(include=['number']).columns.drop(['Survived'])
X = df[feature_cols]
y = df['Survived']

print('Features used:', list(X.columns))

# Plot one row as a heatmap of the numeric feature values.
row_values = X.iloc[5].to_numpy()
feature_df = pd.DataFrame([row_values], columns=X.columns)
sns.heatmap(feature_df, cmap='viridis', cbar=True)
plt.tight_layout()
plt.savefig(base_dir / 'feature_row_heatmap.png', dpi=200)
plt.close()

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
print(importances)
print(importances.shape)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)
print(importance_df)

sns.heatmap(importance_df.pivot_table(index=['feature'], values='importance').T, annot=True, cmap='viridis')
plt.tight_layout()
plt.savefig(base_dir / 'feature_importance_heatmap.png', dpi=200)
plt.close()