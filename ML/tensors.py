import numpy as np # numpy is a library for numerical computing in Python, providing support for arrays and matrices, along with a collection of mathematical functions to operate on these data structures.

a = np.array(2)

print(a.ndim) # returns the number of dimensions of the array

b = np.array([1, 2, 3, 4, 5]) # creates a 1-dimensional array with elements 1 to 5

print(b.ndim) # returns the number of dimensions of the array

c = np.array([[1, 2, 3], [4, 5, 6]]) # creates a 2-dimensional array (matrix) with two rows and three columns

print(c.ndim) # returns the number of dimensions of the array

d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) # creates a 3-dimensional array (tensor) with two matrices, each containing two rows and two columns

print(d.ndim) # returns the number of dimensions of the array