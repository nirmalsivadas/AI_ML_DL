import graphviz.backend as be
import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn import tree
from sklearn.datasets import load_diabetes, load_iris

script_directory = sys.path.pop(0)
from dtreeviz import model as dtreeviz_model
sys.path.insert(0, script_directory)

from IPython.display import Image, display_svg, SVG


def render_tree(model, X_train, y_train, **options):
    feature_names = options.pop("feature_names", None)
    target_name = options.pop("target_name", None)
    class_names = options.pop("class_names", None)
    if "X" in options:
        options["x"] = options.pop("X")
    viz_model = dtreeviz_model(
        model,
        X_train,
        y_train,
        feature_names=feature_names,
        target_name=target_name,
        class_names=class_names,
    )
    return viz_model.view(**options)

clas = tree.DecisionTreeClassifier()  
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)


from sklearn.tree import plot_tree
plot_tree(clas)
plt.show()

viz = render_tree(clas, 
               X_train,
               y_train,
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"])
print(viz)

regr = tree.DecisionTreeRegressor(max_depth=1)
diabetes = load_diabetes()

X_train = diabetes.data
y_train = diabetes.target
regr.fit(X_train, y_train)

viz = render_tree(regr,
               X_train,
               y_train,
               target_name='disease progression',
               feature_names=diabetes.feature_names,
               scale=2
              )
print(viz)

X_train = iris.data
y_train = iris.target
viz = render_tree(clas, 
               X_train,
               y_train,
               target_name='species',
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"], 
               scale = 1.5,
               orientation='LR')
print(viz)

clas = tree.DecisionTreeClassifier()  
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)

X = iris.data[np.random.randint(0, len(iris.data)),:]

viz = render_tree(clas, 
               X_train,
               y_train,
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"], 
               X=X)
print(viz)

print(X)

viz = render_tree(clas, 
               X_train,
               y_train,
               target_name='species',
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"], 
               histtype= 'barstacked',
               scale = 1.5,
               orientation='LR',
               show_node_labels=True)
print(viz)

viz = render_tree(clas, 
               X_train,
               y_train,
               target_name='species',
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"], 
               histtype= 'barstacked',
               scale = 1.5,
               orientation='LR',
               fancy=False)
print(viz)

clas = tree.DecisionTreeClassifier()  
iris = load_iris()

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)

X = iris.data[np.random.randint(0, len(iris.data)),:]

viz = render_tree(clas, 
               X_train,
               y_train,
               target_name='species',
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"], 
               X=X,
               show_just_path=True)
print(viz)


classifier_viz_model = dtreeviz_model(
    clas,
    X_train,
    y_train,
    feature_names=iris.feature_names,
    target_name="species",
    class_names=["setosa", "versicolor", "virginica"],
)
print(classifier_viz_model.explain_prediction_path(X))
classifier_viz_model.instance_feature_importance(X)
plt.show()

diabetes_data = load_diabetes(as_frame=True)
df_diabetes = diabetes_data.frame
X, y = df_diabetes[['bmi']], df_diabetes['target']

dt = tree.DecisionTreeRegressor(max_depth=3, criterion="absolute_error")
dt.fit(X, y)

fig = plt.figure()
ax = fig.gca()
regression_viz_model = dtreeviz_model(
    dt, X, y, feature_names=["bmi"], target_name="target"
)
regression_viz_model.rtree_feature_space(features=["bmi"], ax=ax)
plt.show()

X = df_diabetes[['bmi', 's5']]
y = df_diabetes['target']

dt = tree.DecisionTreeRegressor(max_depth=3, criterion="absolute_error")
dt.fit(X, y)

figsize = (6,5)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')

regression_viz_model = dtreeviz_model(
    dt, X, y, feature_names=["bmi", "s5"], target_name="target"
)
regression_viz_model.rtree_feature_space3D(
    fontsize=14,
    elev=20,
    azim=25,
    dist=8.2,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
