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

print(arr1.ravel()) # flattens the array into 1D
print(arr1.reshape(6, 4)) # reshapes the array
print(arr1.transpose()) # transposes the array
arr2 = np.arange(24).reshape(4, 6)

np.hstack((arr1, arr2)) # stacks the arrays
np.vstack((arr1, arr2))

np.hsplit(arr1, 2)
np.vsplit(arr1, 2)

print(arr1[[1, 2, 3]]) # fancy indexing
