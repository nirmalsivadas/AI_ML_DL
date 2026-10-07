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

lista = range(100)
arr9 = np.arange(100)

import sys

print(sys.getsizeof(arr9))
print(sys.getsizeof(lista))

print(arr9.size * arr9.itemsize)

import time
a = range(1000000)
b = range(1000000, 2000000)
start_time = time.time()

c = [x+y for x, y in zip(a, b)]
print(time.time() - start_time)

start_time = time.time()
c = np.arange(1000000) + np.arange(1000000, 2000000)
print(time.time() - start_time)