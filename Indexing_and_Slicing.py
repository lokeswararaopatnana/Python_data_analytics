# importing numpy
import numpy as np 

# Indexing and Slicing - 1D Arrays:-

# Accessing Elements of a 1D Array:-
    
# Indexing in NumPy Arrays is similar to Python Lists.
    # Zero-based indexing: Index of the first element is '0'.
    # Supports Negative indexing: Index of the last element is '-1'.
"""
numbers_array = np.array([1,3,10,0,2,15,7,100])
third_number = numbers_array[2]
print("third number:",third_number)

fourth_number_from_last = numbers_array[-4]
print("fourth number from last:",fourth_number_from_last)

"""
# Slicing 1D Arrays:-

# Slice with the "[start:stop]" syntax outputs elements in the interval "[start,stop)".
# Slice with the "[start:stop:step]" syntax outputs elements in the interval "[start,stop)" with a step size of 'step'.
"""
numbers_array = np.array([1,3,10,0,2,15,7,100])
sliceed_array = numbers_array[4:7]
print("sliced array:",sliceed_array)

"""
# Using Negative Indices:-
"""
numbers_array = np.array([1,3,10,0,2,15,7,100])
sliced_array = numbers_array[1:-2]
print("sliced array with -ve index:",sliced_array)

"""
# Slicing with step size and negative indices:-
"""
numbers_array = np.array([1,3,10,0,2,15,7,100])
sliced_array = numbers_array[1:-2:2]
print("sliced array with -ve index & step size:",sliced_array)

"""
# Omitting the Start and Stop Indices:-
"""
numbers_array = np.array([1,3,10,0,2,15,7,100])
sliced_array = numbers_array[:4]
print("sliced array with start index omitted:",sliced_array)

sliced_array = numbers_array[-3:]
print("sliced array with end index omitted:",sliced_array)

numbers_array = np.array([1,3,10,0,2,15,7,100])
sliced_array = numbers_array[1::2]
print("sliced array with step size & end index omitted:",sliced_array)

sliced_array = numbers_array[::2]
print("sliced array with step size & start,end index omitted:",sliced_array)

"""
# Slicing with a Negative Step Size:-
"""
numbers_array = np.array([1,3,10,0,2,15,7,100])
sliced_array = numbers_array[-1:0:-2]
print("sliced array with -ve step size:",sliced_array)

sliced_array = numbers_array[3:-1:-1]
print("sliced array with -ve step size,stop to right of start:",sliced_array)

sliced_array = numbers_array[-2:1:2]
print("sliced array with +ve step size, stop to the left from start:",sliced_array)

"""
# ---------------------------------------------------------
# Modifying Slices:-
# In NumPy, slices can be assigned
    # a constant or
    # an array of the right shape
"""
numbers_array = np.array([1,3,10,0,2,15,7,100])
numbers_array[1:4] = 5
print("numbers array with 5s:",numbers_array)

numbers_array[2:6] = np.array([1,2,3,4])
print("numbers array with updated slice:",numbers_array)

"""

# In NumPy slicing returns a view of the original array, instead of creating a new array.
    # This makes slicing numpy arrays very fast.
    # This also means modifying a slice, modifies the underlying array as well.
"""    
numbers_array = np.array([1,3,5,7,9,11,13,15,17,19])
sliced_array = numbers_array[1:4]
sliced_array[0] = -1
print("sliced array after modifying:",sliced_array)
print("numbers array after modifying the slice:",numbers_array)

"""
# Assignment also returns a view of the original array:-
"""
numbers_array = np.array([1,3,5,7,9,11,3,15,17])
new_array = numbers_array

print("numbers array before modifying the new array:",numbers_array)

new_array[0] = 1202
new_array[1] = 2000

print("new array modification:",new_array)

print("numbers array after modifying the new array:",numbers_array)

"""
# ---------------------------------------------------------

# Copying NumPy Arrays using ".copy()":-
"""
numbers_array = np.array([2,4,6,8,10,12,14,16])
new_array = numbers_array.copy()
new_array[0] = 13
new_array[1] = 57
print("new array after modification:",new_array)
print("numbers array after modifying the new array:",numbers_array)

"""
# ---------------------------------------------------------
# Indexing and Slicing - 2D Arrays:-

# Accessing Elements of a 2D Array:-
"""
array_2d = np.array([['A','B','C','D'],
                     ['E','F','G','H'],
                     ['I','J','K','L']])
print("[0,0]:",array_2d[0,0])
print("[1,-1]:",array_2d[1,-1])

# Accessing a single row or a single column:-

# Accessing a single row:-
first_row = array_2d[0]
print("first row:",first_row)

second_row = array_2d[1]
print("second row:",second_row)

third_row = array_2d[2]
print("third row:",third_row)

# Accessing a single column:-
first_col = array_2d[:,0]
print("first col:",first_col)

"""
# ---------------------------------------------------------

# Slicing 2D Arrays:-
# Slicing rows:-
"""
array_2d = np.array([['A','B','C','D'],
                     ['E','F','G','H'],
                     ['I','J','K','L']])
print(array_2d[1:3])

# Slicing rows and columns:-
print(array_2d[1:3,2:4])

# Omitting the start and stop indices can be done in 2D Arrays also:-
print(array_2d[1:,2:])

"""
# ---------------------------------------------------------
# Slicing with Step Size:-
# Slicing with a +ve step size:-
"""
array_2d = np.array([['A','B','C','D'],
                     ['E','F','G','H'],
                     ['I','J','K','L']])
print(array_2d[:,0:4:2])

# Slicing with a -ve step size:-

print(array_2d[-1:-4:-2,:])

"""
# ---------------------------------------------------------
# Modifying slices of a 2D Array:-
"""
array_2d = np.array([['A','B','C','D'],
                     ['E','F','G','H'],
                     ['I','J','K','L']])
sliced_array = array_2d[1:3,:]
sliced_array[1] = 'O'
print("sliced array after modification:",sliced_array)
print("array_2d after modifying with the slice:",array_2d)

"""
# ---------------------------------------------------------
# Advanced Indexing and Slicing:-

# Fancy Indexing:-
# NumPy arrays can also be indexed with other arrays.
    # The result will be a new array which contains elements at the indices given in the input array.
    
"""
numbers_array = np.array([2,4,6,8,10,12,14,16,18,20])
index_array = np.array([1,4,3])
indexed_array = numbers_array[index_array]
print("indexed array:",indexed_array)

numbers_array = np.array([1,3,5,7,9,11,13,15,17,19])
index_array = np.array([1,-2,-5,0,-3])
indexed_array = numbers_array[index_array]
print("indexed array:",indexed_array)

"""
# The index array selects the elements at the given indices from the original array.
# The index array should only contain integers that are in the range of index.
# The shape of the index array doesn't beed to match the shape of the original array.

# ---------------------------------------------------------

# Boolean/Mask Index Arrays:-

# Boolean arrays can also be used as index arrays.
"""
numbers_array = np.array([101,102,103,104,105])
mask_array = np.array([True,False,True,True,False])
indexed_array = numbers_array[mask_array]
print("indexed array:",indexed_array)

"""
# The output will contain only the elements whose corresponding value in the 'mask_array' is 'True'.
# Unlike normal index arrays, the shape of Boolean Index Arrays should match the shape of the original array.

# ---------------------------------------------------------
# Fancy Indexing and Slicing in 2D Arrays:-
"""
array_2d = np.array([['A','B','C'],
                     ['D','E','F'],
                     ['G','H','I']])
row_1 = array_2d[(0,2),1]
print(row_1)

row_2 = array_2d[[0,2],2]
print(row_2)

row_3 = array_2d[[1,2]]
print(row_3)

col_1 = array_2d[0,(0,1)]
print(col_1)

col_2 = array_2d[1,[0,2]]
print(col_2)

res_1 = array_2d[[0]]
print(res_1)

res_2 = array_2d[[0,2]]
print(res_2)

res_3 = array_2d[[0,1]][:,[1,2]]
print(res_3)

res_4 = array_2d[[1,2],[0,1]]
print(res_4)

"""
# ---------------------------------------------------------
# Masking in 2D Arrays:-
"""
array_2d = np.array([['A', 'B', 'C', 'D', 'E'],
                    ['F', 'G', 'H', 'I', 'J'],
                    ['K', 'L', 'M', 'N', 'O'],
                    ['P', 'Q', 'R', 'S', 'T'],
                    ['U', 'V', 'W', 'X', 'Y']])
row_mask = np.array([False,True,True,False,True])
print(array_2d[row_mask,3])

row_mask = np.array([True,False,True,False,True])
print(array_2d[row_mask,:])

col_mask = np.array([True,False,False,True,False])
print(array_2d[3,col_mask])

"""