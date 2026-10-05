from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_regression

OUTPUT_DIR = Path(__file__).resolve().parent / "files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X, y = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=20,
    random_state=13,
)

plt.scatter(X, y)
plt.savefig(OUTPUT_DIR / "scatter_plot.png", bbox_inches="tight")
plt.close()

m_arr = np.linspace(-150, 150, 10)
b_arr = np.linspace(-150, 150, 10)
mGrid, bGrid = np.meshgrid(m_arr, b_arr)

final = np.vstack((mGrid.ravel().reshape(1, 100), bGrid.ravel().reshape(1, 100))).T

z_arr = []

for i in range(final.shape[0]):
    z_arr.append(np.sum((y - final[i, 0] * X.reshape(100) - final[i, 1]) ** 2))

z_arr = np.array(z_arr).reshape(10, 10)

fig = go.Figure(data=[go.Surface(x=m_arr, y=b_arr, z=z_arr)])
fig.update_layout(
    title="Cost Function",
    autosize=False,
    width=500,
    height=500,
    margin=dict(l=65, r=50, b=65, t=90),
)
fig.write_html(OUTPUT_DIR / "cost_function.html")

b = 150
m = -127.82
lr = 0.001
all_b = []
all_m = []
all_cost = []

epochs = 30

for i in range(epochs):
    slope_b = 0
    slope_m = 0
    cost = 0
    for j in range(X.shape[0]):
        slope_b = slope_b - 2 * (y[j] - (m * X[j]) - b)
        slope_m = slope_m - 2 * (y[j] - (m * X[j]) - b) * X[j]
        cost = cost + (y[j] - m * X[j] - b) ** 2

    b = b - (lr * slope_b)
    m = m - (lr * slope_m)
    all_b.append(b)
    all_m.append(m)
    all_cost.append(cost)

fig = px.scatter_3d(
    x=np.array(all_m).ravel(),
    y=np.array(all_b).ravel(),
    z=np.array(all_cost).ravel() * 100,
)
fig.add_trace(go.Surface(x=m_arr, y=b_arr, z=z_arr * 100))
fig.write_html(OUTPUT_DIR / "cost_function2.html")

fig = go.Figure(
    go.Scatter(
        x=np.array(all_m).ravel(),
        y=np.array(all_b).ravel(),
        name="Gradient descent path",
        line=dict(color="#fff", width=4),
    )
)
fig.add_trace(go.Contour(z=z_arr, x=m_arr, y=b_arr))
fig.write_html(OUTPUT_DIR / "gradient_path.html")

fig, ax = plt.subplots(1, 1, figsize=(18, 4))
cp = ax.contourf(m_arr, b_arr, z_arr)
ax.plot(np.array(all_m).ravel(), np.array(all_b).ravel(), color="white")
fig.colorbar(cp)
ax.set_title("Filled Contours Plot")
ax.set_xlabel("m")
ax.set_ylabel("b")
fig.savefig(OUTPUT_DIR / "filled_contours.png", bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(9, 5))
axis = plt.axes(xlim=(-150, 150), ylim=(-150, 150))
axis.contourf(m_arr, b_arr, z_arr)
line, = axis.plot([], [], lw=2, color="white")

xdata, ydata = [], []


def animate(i):
    label = "epoch {0}".format(i + 1)
    xdata.append(all_m[i])
    ydata.append(all_b[i])
    line.set_data(xdata, ydata)
    axis.set_xlabel(label)
    return line,


anim = animation.FuncAnimation(fig, animate, frames=30, repeat=False, interval=500)
anim.save(OUTPUT_DIR / "gradient_descent_animation.gif", writer="pillow", fps=2)
plt.close(fig)

