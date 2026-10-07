import numpy as np

arr1 = np.array([[1,2,3],[4,5,6]])
arr2 = np.array(1)

print(arr1+arr2) # broadcasting -> will add 1 to all elements of arr1

arr3 = np.array([1,2,3])
arr4 = np.array([[1,2,3],[4,5,6]])

print(arr3+arr4)

arr5 = np.array([[1,2],[2,1],[21,2],[1,2]])
arr6 = np.array([[1,2,3],[4,5,6]])

print(arr5+arr6)

print(np.random.random())
np.random.seed(42)
print(np.random.random())
print(np.random.uniform(1,2))

a = np.random.randint(1,10,(3,3))
b = np.random.randint(1,10,(3,3))
print(a)
print(b)
print(a+b)
print(np.max(a))
print(np.max(a, axis=0))
print(np.max(a, axis=1))
print(np.min(a))
print(np.min(a, axis=0))
print(np.min(a, axis=1))
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
print(np.mean(a))
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))
print(np.std(a))
print(np.std(a, axis=0))
print(np.std(a, axis=1))
print(np.var(a))
print(np.var(a, axis=0))
print(np.var(a, axis=1))
print(np.sin(a))
print(np.cos(a))
print(np.tan(a))
print(np.log(a))
print(np.exp(a))

print(np.where(a>5, a, 0))

print(np.percentile(a, 50))
print(np.percentile(a, 50, axis=0))
print(np.percentile(a, 50, axis=1))
print(np.sort(a))
