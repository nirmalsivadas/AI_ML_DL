import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'files', 'student_clustering.csv')

df = pd.read_csv(dataset_path)
X = df.iloc[:, :].values

km = KMeans(n_clusters=4, max_iter=500, random_state=42)
y_means = km.fit_predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], color='red')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], color='blue')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], color='green')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], color='yellow')
plt.title('KMeans Clustering on Student Dataset')
plt.show()