from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
X,y = make_classification(n_samples=5, n_classes=2,
                               n_features=2, n_informative=2, n_redundant=0,
                               random_state=0)

clf = DecisionTreeClassifier()
clf.fit(X,y)

from sklearn.tree import plot_tree
plot_tree(clf)
plt.show()

print(clf.feature_importances_)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=2)
rf.fit(X,y)

print(rf.feature_importances_)

print(rf.estimators_[0].feature_importances_)
print(rf.estimators_[1].feature_importances_)

print((1 + 0.555)/2)

X,y = make_classification(n_samples=15, n_classes=2,
                               n_features=2, n_informative=2, n_redundant=0,
                               random_state=0)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X,y)

from sklearn.tree import plot_tree
plot_tree(clf)
plt.show()

print(clf.feature_importances_)