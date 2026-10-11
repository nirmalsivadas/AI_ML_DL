import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

np.random.seed(42)
X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

def plot_decision_boundary(clf):
    plt.figure(figsize=(12, 8))
    x_range = np.linspace(X.min(), X.max(), 100)
    xx1, xx2 = np.meshgrid(x_range, x_range)
    y_hat = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    y_hat = y_hat.reshape(xx1.shape)
    plt.contourf(xx1, xx2, y_hat, alpha=0.2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title("AdaBoost Classifier")
    plt.show()


abc = AdaBoostClassifier()
cv_score = np.mean(cross_val_score(abc, X, y, scoring='accuracy', cv=10))
print(f"Baseline cross-validation accuracy: {cv_score:.4f}")
abc.fit(X, y)
plot_decision_boundary(abc)

abc = AdaBoostClassifier(n_estimators=1500, learning_rate=0.1)
abc.fit(X, y)
plot_decision_boundary(abc)

param_grid = {
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
}

grid_search = GridSearchCV(
    estimator=AdaBoostClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    cv=10,
    scoring='accuracy',
)

grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Link
# https://stats.stackexchange.com/questions/82323/shrinkage-parameter-in-adaboost