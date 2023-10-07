# importing numpy
import numpy as np

# Padding with different pad widths:-

# "numpy.pad(array,pad_with,mode = 'constant',**kwargs)".
# --> 'pad_width' is the number of values padded to the edges of each 'axis'.
# --> Output is padded array of rank to 'array', with shape increased according to 'pad_width'.
"""
array_2D = np.array([['A','B','C'],
                     ['D','E','F'],
                     ['G','H','I']])
"""
# Padding with 'constant' mode.
"""
res_pad = np.pad(array_2D,pad_width=((1,2),(3,1)),mode='constant',constant_values=(('X','Y'),('W','Z')))
print(res_pad)
"""
# Pad with 'edge' mode.
"""
res_edge = np.pad(array_2D,((1,2),(3,1)),'edge')
print(res_edge)
"""
# ---------------------------------------------------------

# More Numpy Methods:-

# np.unique:-
# "np.unique(arr,return_index = False,return_inverse = False,return_counts = False,axis = None)".
# --> Returns the sorted unique elements of an array.
"""
a = np.array([1,3,3,1,2,1])
print(np.unique(a))
"""
# If 'return_index = True', it returns the indices of the occurrences of the unique values in the original array, along with the unique sorted array.
"""
unique_array, first_indices = np.unique(a,return_index = True)
print(unique_array,first_indices)
"""
# If 'return_counts = True', it returns the number of times each of the unique values comes up in the original array.
"""
a = np.array([1,3,3,1,2,1])
print(np.unique(a,return_counts=True))
"""
# The following code returns the unique rows of a 2D array.
"""
a = np.array([[1,2,1],
              [1,2,1],
              [3,2,4]])
print(np.unique(a,axis=0))
"""
# If 'return_inverse' = True, it returns the indices to reconstruct the original array from the unique array.
"""
a = np.array([1,3,3,1,2,1])
unique_array,indices = np.unique(a,return_inverse=True)
print("original array:\n",a)
print("indices:\n",indices)
print("reconstructed orginal array using fancy indexing:\n",unique_array)
"""

# np.where:-

# "numpy.where(condition[,x,y])".
# --> For all the elements where the condition is 'True', the returned array has the corresponding elements chosen from 'x'.
# --> For all the elements where the condition is 'False', the returned array has the corresponding elements chosen form 'y'.
"""
a = np.arange(5)
res1 = np.where(a > 2,[11,12,13,14,15],[111,222,333,444,555])
print(res1)

a = np.arange(10)
res2 = np.where(a > 7,0,100)
print(res2)

a = np.arange(10)
res3 = np.where(a > 5,a*2,a*a)
print(res3)

a = np.array([[1,2,3],
              [4,5,6]])
res4 = np.where(a < 5, a,0)
print(res4)
"""
# Comparison can also be done between two different arrays
"""
a = np.array([[1,2,3,],
              [4,5,6]])
b = np.ones([2,3]) * 4
print("a:\n",a)
print("b:\n",b)
print("np.where result:\n",np.where(a < b,a,0))
"""
# When the two arrays are of different shapes, they are broadcast together before applying the operation.
"""
x = np.array([[0],
              [1],
              [2]])
y = np.array([[3,4,5,6]])
print(np.where(x < y, x,10+y))
"""

# np.any:-

# "numpy.any(a,axis = None)".
# --> Tests whether 'any' of the array elements along the given 'axis' evaluates to 'True'.
# --> Returns a new boolean ndarray.
"""
a = np.array([False,True,True,False])
print(np.any(a))

a = np.array([1,0,0,2])
print(np.any(a))

a = np.array([0,0,0,0])
print(np.any(a))

a = np.array([[1,0,3,4],
              [2,0,6,5]])
res_array_0 = np.any(a,axis=0)
print("result array along axis 0:\n",res_array_0)

res_array_1 = np.any(a,axis=1)
print("result array along axis 0:\n",res_array_1)

res_array_none = np.any(a)
print("result array along axis 0:\n",res_array_none)
"""

# np.all:-

# "numpy.all(a,axis = None,out = None)".
# --> Tests whether 'all' of the array elements along the given 'axis' evaluate to 'True'.
# --> Returns a new boolean ndarray.
"""
a = np.array([1,2,3,0])
res_any = np.any(a)
res_all = np.all(a)
print("res any:",res_any)
print("res all:",res_all)

a = np.array([[1,0,3,4],
              [2,0,6,5]])
res_array_0 = np.all(a,axis=0)
print("result array along axis 0:\n",res_array_0)

res_array_1 = np.all(a,axis=1)
print("result array along axis 1:\n",res_array_1)

res_array_none = np.all(a)
print("result array along axis none:\n",res_array_none)
"""
# ---------------------------------------------------------

# np.nan:-

# "numpy.nan".
# --> IEEE 754 floating point representation of Not a Number(NaN).
"""
a = np.array([0]) / 0
print(a)
"""
# We can not compare two NaN's.
"""
print(np.nan == np.nan)
"""
# ---------------------------------------------------------

# np.inf:-

# "numpy.inf".
# --> IEEE 754 floating representation of (positive) infinity.
"""
a = np.array([1]) / 0
print(a)
"""
# Comparing two inf numbers.
"""
print(np.inf == np.inf)
"""