def dy_dx(x):
  return 2*x

dy_dx(3)
import torch
x = torch.tensor(3.0, requires_grad=True)

y = x**2
x
y
y.backward()
x.grad
import math

def dz_dx(x):
    return 2 * x * math.cos(x**2)

dz_dx(4)
x = torch.tensor(4.0, requires_grad=True)
y = x ** 2
z = torch.sin(y)

x
y
z
z.backward()
x.grad
y.grad

import torch

# Inputs
x = torch.tensor(6.7)  # Input feature
y = torch.tensor(0.0)  # True label (binary)

w = torch.tensor(1.0)  # Weight
b = torch.tensor(0.0)  # Bias

# Binary Cross-Entropy Loss for scalar
def binary_cross_entropy_loss(prediction, target):
    epsilon = 1e-8  # To prevent log(0)
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))

# Forward pass
z = w * x + b  # Weighted sum (linear part)
y_pred = torch.sigmoid(z)  # Predicted probability

# Compute binary cross-entropy loss
loss = binary_cross_entropy_loss(y_pred, y)
loss

# Derivatives:
# 1. dL/d(y_pred): Loss with respect to the prediction (y_pred)
dloss_dy_pred = (y_pred - y)/(y_pred*(1-y_pred))

# 2. dy_pred/dz: Prediction (y_pred) with respect to z (sigmoid derivative)
dy_pred_dz = y_pred * (1 - y_pred)

# 3. dz/dw and dz/db: z with respect to w and b
dz_dw = x  # dz/dw = x
dz_db = 1  # dz/db = 1 (bias contributes directly to z)

dL_dw = dloss_dy_pred * dy_pred_dz * dz_dw
dL_db = dloss_dy_pred * dy_pred_dz * dz_db

print(f"Manual Gradient of loss w.r.t weight (dw): {dL_dw}")
print(f"Manual Gradient of loss w.r.t bias (db): {dL_db}")

x = torch.tensor(6.7)
y = torch.tensor(0.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

w
b
z = w*x + b
z

y_pred = torch.sigmoid(z)
y_pred

loss = binary_cross_entropy_loss(y_pred, y)
loss

loss.backward()
print(w.grad)
print(b.grad)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x

y = (x**2).mean()
y

y.backward()
x.grad

# clearing grad
x = torch.tensor(2.0, requires_grad=True)
x

y = x ** 2
y

y.backward(retain_graph=True)
x.grad

x.grad.zero_()
# disable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
x

y = x ** 2
y

y.backward(retain_graph=True)
x.grad

# option 1 - requires_grad_(False)
# option 2 - detach()
# option 3 - torch.no_grad()

x.requires_grad_(False)
x

y = x ** 2
y
y.backward()
x = torch.tensor(2.0, requires_grad=True)
x

z = x.detach()
z

y = x ** 2
y

y1 = z ** 2
y1

y.backward()
y1.backward()

x = torch.tensor(2.0, requires_grad=True)
x

y = x ** 2
y
y.backward()