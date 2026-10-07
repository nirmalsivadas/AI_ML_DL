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