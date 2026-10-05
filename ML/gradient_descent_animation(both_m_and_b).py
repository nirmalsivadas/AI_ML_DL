from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
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

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X, y)
x_i = np.arange(-3, 3, 0.1)
line, = ax.plot(x_i, x_i * 50 - 4, "r-", linewidth=2)
fig.savefig(OUTPUT_DIR / "initial_regression_fit.png", bbox_inches="tight")
plt.close(fig)

b = -520
m = 600
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


def save_animation(output_path, y_values, y_limits, x_label, title):
    fig = plt.figure(figsize=(9, 5))
    axis = plt.axes(xlim=(0, epochs), ylim=y_limits)
    line, = axis.plot([], [], lw=2)
    xdata, ydata = [], []

    def animate(i):
        label = f"epoch {i + 1}"
        xdata.append(i)
        ydata.append(y_values[i])
        line.set_data(xdata, ydata)
        axis.set_xlabel(label)
        axis.set_title(title)
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=epochs, repeat=False, interval=500)
    anim.save(output_path, writer="pillow", fps=2)
    plt.close(fig)


save_animation(
    OUTPUT_DIR / "animation5_cost.gif",
    all_cost,
    (0, max(all_cost) * 1.1),
    "Epoch",
    "Cost over epochs",
)
save_animation(
    OUTPUT_DIR / "animation6_b.gif",
    all_b,
    (min(all_b) * 1.1, max(all_b) * 1.1),
    "Epoch",
    "Intercept b over epochs",
)
save_animation(
    OUTPUT_DIR / "animation7_m.gif",
    all_m,
    (min(all_m) * 1.1, max(all_m) * 1.1),
    "Epoch",
    "Slope m over epochs",
)

# Combined regression line animation
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X, y)
x_i = np.arange(-3, 3, 0.1)
line, = ax.plot(x_i, x_i * 50 - 4, 'r-', linewidth=2)


def update(i):
    label = f"epoch {i + 1}"
    line.set_ydata(x_i * all_m[i] + all_b[i])
    ax.set_xlabel(label)
    ax.set_title("Regression line update")
    return line,


anim = animation.FuncAnimation(fig, update, repeat=False, frames=epochs, interval=500)
anim.save(OUTPUT_DIR / "animation4_regression_line.gif", writer="pillow", fps=2)
plt.close(fig)
