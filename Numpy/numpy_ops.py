import numpy as np

arr1 = np.array([1,2,3,4,5,6])
arr2 = np.array([4,5,6,7,8,9])

print(arr1-arr2)
print(arr1+arr2)
print(arr1*arr2)
print(arr1/arr2)

print(arr1*2)
print(arr1>3) # boolean
arr3 = np.arange(6).reshape(2,3)
arr4 = np.arange(6,12).reshape(3,2)

print(arr3.dot(arr4))
print(arr4.max)
print(arr4.min(axis=1))
print(arr4.argmax)
print(arr4.argmin)
print(arr4.sum(axis=0))
print(arr4.sum(axis=1))
print(arr4.mean(axis=1))
print(arr4.std(axis=1))
print(arr4.var(axis=1))
print(np.sin(arr1))
print(np.cos(arr1))
print(np.tan(arr1))
print(np.log(arr1))
print(np.exp(arr1))
