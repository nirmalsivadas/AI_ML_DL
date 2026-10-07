import numpy as np
import matplotlib.pyplot as plt
arr1 = np.random.randint(low=0, high=10, size=(3, 3))
print(arr1[arr1>5]) # indexing with boolean array

arr2 = np.linspace(-40,40,100)
arr3 = np.sin(arr2)

plt.plot(arr2, arr3)
plt.show() 
