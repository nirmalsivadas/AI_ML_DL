import numpy as np

arr1 = np.arange(24).reshape(4, 6)
print(arr1)

print(arr1[1, 2])
print(arr1[1, :])
print(arr1[:, 2])
print(arr1[1:3, 2:4])

for row in arr1:
  print(row)

for element in np.nditer(arr1):
  print(element)