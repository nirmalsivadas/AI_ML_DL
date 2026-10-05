from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

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

reg = LinearRegression()
reg.fit(X, y)
print(reg.coef_)
print(reg.intercept_)

b = -150
m = 27.82
lr = 0.001
all_b = []
all_cost = []
epochs = 30

for _ in range(epochs):
    slope = 0
    cost = 0
    for j in range(X.shape[0]):
        error = y[j] - (m * X[j]) - b
        slope -= 2 * error
        cost += error**2

    b -= lr * slope
    all_b.append(b)
    all_cost.append(cost)

all_b = np.asarray(all_b).ravel()
all_cost = np.asarray(all_cost).ravel()
num_epochs = np.arange(epochs)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(X, y)
x_i = np.arange(-3, 3, 0.1)
line, = ax.plot(x_i, x_i * m + all_b[0], "r-", linewidth=2)


def update_regression(i):
    ax.set_xlabel(f"epoch {i + 1}")
    line.set_ydata(x_i * m + all_b[i])
    return (line,)


regression_animation = animation.FuncAnimation(
    fig, update_regression, frames=epochs, repeat=False, interval=500
)
regression_animation.save(
    OUTPUT_DIR / "animation0_regression_line.gif",
    writer="pillow",
    fps=2,
)
fig.savefig(OUTPUT_DIR / "regression_data.png", bbox_inches="tight")
plt.close(fig)


def save_series_animation(values, output_path, title, y_limits):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, epochs)
    ax.set_ylim(*y_limits)
    ax.set_title(title)
    line, = ax.plot([], [], lw=2)
    x_data = []
    y_data = []

    def animate(i):
        x_data.append(num_epochs[i])
        y_data.append(values[i])
        line.set_data(x_data, y_data)
        ax.set_xlabel(f"epoch {i + 1}")
        return (line,)

    result = animation.FuncAnimation(
        fig, animate, frames=epochs, repeat=False, interval=500
    )
    result.save(output_path, writer="pillow", fps=2)
    plt.close(fig)


save_series_animation(
    all_cost,
    OUTPUT_DIR / "animation1_cost.gif",
    "Cost over epochs",
    (0, float(all_cost.max()) * 1.1),
)
save_series_animation(
    all_b,
    OUTPUT_DIR / "animation2_intercept.gif",
    "Intercept b over epochs",
    (
        float(all_b.min() - max(np.ptp(all_b) * 0.05, 1)),
        float(all_b.max() + max(np.ptp(all_b) * 0.05, 1)),
    ),
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(all_b, all_cost)
point = ax.scatter([], [], color="red", marker="+")
ax.set_xlim(float(all_b.min()) - 10, float(all_b.max()) + 10)
ax.set_ylim(0, float(all_cost.max()) * 1.1)
ax.set_xlabel("Intercept b")
ax.set_ylabel("Cost")
ax.set_title("Gradient descent path")


def update_path(i):
    point.set_offsets(np.array([[all_b[i], all_cost[i]]]))
    ax.set_xlabel(f"epoch {i + 1} — intercept b")
    return (point,)


path_animation = animation.FuncAnimation(
    fig, update_path, frames=epochs, interval=500, repeat=False
)
path_animation.save(
    OUTPUT_DIR / "animation3_cost_vs_intercept.gif",
    writer="pillow",
    fps=2,
)
plt.close(fig)

b_input = np.linspace(-150, 150, 100)
cost_input = np.sum(
    (y[:, np.newaxis] - m * X - b_input[np.newaxis, :]) ** 2,
    axis=0,
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(b_input, cost_input)
ax.set_xlabel("Intercept b")
ax.set_ylabel("Cost")
ax.set_title("Cost function by intercept")
fig.savefig(OUTPUT_DIR / "cost_function_by_intercept.png", bbox_inches="tight")
plt.close(fig)
