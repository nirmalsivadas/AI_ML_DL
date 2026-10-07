import numpy as np

arr1 = np.array([1, 2, 3])
print(arr1)

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)

arr3 = np.copy(arr2)
print(arr3)

arr4 = np.arange(1, 10)
print(arr4)

arr5 = np.arange(1, 10, 2)
print(arr5)

arr6 = np.zeros((3, 4))
print(arr6)

arr7 = np.ones((3, 4))
print(arr7)

arr8 = np.identity(3)
print(arr8)

print(arr1.shape)
print(arr2.shape)
print(arr3.shape)
print(arr4.shape)
print(arr5.shape)
print(arr6.shape)
print(arr7.shape)
print(arr8.shape)
print(arr8.ndim)
print(arr8.dtype)
print(arr8.itemsize)
print(arr8.size)
arr8.astype('float64')
print(arr8)