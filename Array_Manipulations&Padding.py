# importing numpy
import numpy as np

# Broadcasting with different dimensions:-
""" 
# Two arrays do not need to have the same number of dimensions to do element-wise operations.
# When the dimensions are not the same, to check the compatibility for broadcasting, start with the trailing dimensions and check the compatibility for each dimension moving towards the left.
# If all the existing  dimensions are compatible, then we can do element-wise operations with the two arrays.

    >> a       (2d array): 3 * 2
    >> b       (1d array):     2
    >> a + b   (2d array): 3 * 2

"""
"""
a = np.zeros([3,2])
b = np.ones(2)
print((a + b).shape)
"""
""" 
    >> a       (4d array): 3 * 1 * 4 * 2
    >> b       (3d array):     3 * 4 * 2
    >> a + b   (4d array): 3 * 3 * 4 * 2
"""
"""
a = np.zeros([3,1,4,2])
b = np.ones([3,4,2])
print((a + b).shape)
"""
# ---------------------------------------------------------

# Array Manipulations:-

# Changing the array shape:-

# Reshaping:-
# np.reshape(arr, new_shape) 
# --> Returns a new array with the shape "new_shape".
# --> The shape of the original array will not change.
# --> Might return a 'view' of the original array.
"""
array_1d = np.array([1,3,5,0,2,4,9,7,3,11,5,13])
reshaped_array = np.reshape(array_1d,(4,3))
print("reshaped_array:\n",reshaped_array)
print("array_1d:\n",array_1d)
"""
# "ndarray.reshape(newshape)" is equivalent to "np.reshape()".
"""
array_1d = np.array([1,3,5,0,2,4,9,7,3,11,5,13])
reshaped_array = array_1d.reshape((3,4))
print("reshaped_array:\n",reshaped_array)
print("array_1d:",array_1d)
"""
# Using "-1" as a placeholder in the "reshape()" method.
"""
array_1d = np.array([1,3,5,0,2,4,9,7,3,11,5,13])

reshaped_array = array_1d.reshape((4,-1))
print("reshaped array with placeholder:\n",reshaped_array)

reshaped_array = array_1d.reshape((2,-1,3))
print("reshaped array with placeholder:\n",reshaped_array)

array_2d = np.array([[1,3,5],
                     [0,2,4]])
print("reshaped to (3,2):\n",array_2d.reshape(3,2))
print("reshaped to 6:\n",array_2d.reshape(6))
"""

# "np.ravel(arr)":-
# --> Returns a contiguous falttened array (A 1D array, containing the elements of the input array).
# --> The shape of the original array will not change.
# --> Returns a 'view' of the original array.
"""
array_2d = np.array([[1,3,5],
                     [0,2,4]])
ravled_array = np.ravel(array_2d)
ravled_array[0] = -10
print("raveled_array:\n",ravled_array)
print("array_2d:\n",array_2d)
"""
# "np.ravel(arr)" is equivalent to "np.reshape(arr,-1) and arr.ravel()".
"""
array_2d = np.array([[1,3,5],
                     [0,2,4]])
print("raveled array:\n",np.ravel(array_2d))
print("raveled array alternative:\n",array_2d.ravel())
print("raveled array:\n",np.reshape(array_2d,-1))
"""

# "ndarray.flatten()":-
# --> Returns a 'copy' of the array flattened to one dimension.
# --> The shape of the original array will not change.
"""
array_2d = np.array([[1,3,5],
                     [0,2,4]])
flattened_array = array_2d.flatten()
flattened_array[0] = 143
print("flattened array:\n",flattened_array)
print("array_2d:\n",array_2d)
"""

# Using "np.newaxis":-
# It is used to increase the dimension of an existing array by one.
# --> "nD" array becomes "(n + 1)D" array.
# It can be used to convert a "1D" array to either a row vector (or) a column vector.
"""
a = np.array([1,3,5])
row_vector = a[np.newaxis,:]
print(row_vector)
print("row vector alternative:\n",a[np.newaxis])
print("shape of the original array:",a.shape)
print("shape of a[newaxis,:]:",row_vector.shape)

a = np.array([1,3,5])
column_vector = a[:,np.newaxis]
print(column_vector)
print("shape of the original array:",a.shape)
print("shape of a[:,newaxis]:",column_vector.shape)

print("shape of a[:,newaxis,np.newaxis]:\n",a[:,np.newaxis,np.newaxis])
print("shape of a[:,newaxis,np.newaxis]:",a[:,np.newaxis,np.newaxis].shape)
print("shape of a[np.newaxis,:,np.newaxis]:\n",a[np.newaxis,:,np.newaxis])
print("shape of a[np.newaxis,:,np.newaxis]:",a[np.newaxis,:,np.newaxis].shape)
"""

# "np.squeeze":-
# "np.squeeze(arr,axis)"
# --> It removes single-dimensional entries from the shape of an array.
"""
a = np.array([[[1],[3],[5]]])
print(a)
print(a.shape)
print(np.squeeze(a))
print(np.squeeze(a).shape)

a = np.array([[[1],[3],[5]]])
print(np.squeeze(a,axis=2))
print(np.squeeze(a,axis=2).shape)
"""
# The following code gives an 'error' as the size of the selecated axis is not equal to one.
"""
print(np.squeeze(a,axis=1).shape)
"""
# When expanded arrays are squeezed, the original array is obtained.
"""
a = np.array([1,3,5])
print("original:",a)

expanded = np.expand_dims(a,axis=0)
print("expanded:",expanded)

squeezed = np.squeeze(expanded,axis=0)
print("squeezed:",squeezed)

a = np.array([1,3,5])
print("original:",a)

expanded = np.expand_dims(a,axis=1)
print("expanded:",expanded)

squeezed = np.squeeze(expanded,axis=1)
print("squeezed:",squeezed)
"""
# ---------------------------------------------------------

# Change the axes of an array:-

# "np.moveaxis":-
# "np.moveaxis(arr,original_positions,new_positions)"
# --> Returns the view of an array with the axes moved from their original positions to the new positions.
# --> Other axes remain in their original order.
"""
array_5d = np.ones((2,3,4,5,6))
axis_moved_array = np.moveaxis(array_5d,[0,1],[1,2])
print(axis_moved_array.shape)

array_3d = np.array([[[5,5,5],
                      [0,0,0]],
                     [[1,2,3],
                      [-1,-2,-3]],
                     [[6,7,8],
                      [-6,-7,-8]]])
print("original array:\n",array_3d)
moved_axis_array = np.moveaxis(array_3d,0,2)
print("array after moving axes:\n",moved_axis_array)
"""

# "np.swapaxes":-
# "np.swapaxes(arr,axix1,axis2)".
# --> Returns an array with axis1 and axis2 interchanged.
# --> Other axes remain in their original order.
"""
array_3d = np.array([[[5,5,5],
                      [0,0,0]],
                     [[1,2,3],
                      [-1,-2,-3]],
                     [[6,7,8],
                      [-6,-7,-8]]])
print("original array:\n",array_3d)
swaped_axes_array = np.swapaxes(array_3d,1,2)
print("array after swaping axes:\n",swaped_axes_array)
"""
# ---------------------------------------------------------

# Splitting an array:-
# "np.split(arr,indices_or_sections,axis=0)".
# --> Splits the given array 'arr' into mupltiple sub-arrays along the given 'axis' based on 'indices_or_sections' and returns a list of sub-arrays.
# If 'indices_or_sections' is an integer say 'N', then 'arr' will be divided into 'N' equal arrays along the given 'axis'.
"""
array_1d = np.array([1,7,11,0,3,17])
split_arrays = np.split(array_1d,2)
print(split_arrays)
"""
# If 'indices_or_sections' is a lsit, then 'arr' will be split into sub-arrays at the indices mentioned in the list.
"""
array_1d = np.array([7,7,11,0,3,17])
split_arrays = np.split(array_1d,[1])
print(split_arrays)
"""
# Helper function to print splits of an array.

def print_splits(array_to_split):
    for item in array_to_split:
        print(item)

# Split an array along given axis:-
"""
# array_1d = np.array([[1,7,11,12],
#                      [0,3,17,2]])
# split_arrays1 = np.split(array_1d,2,axis=1)
# print_splits(split_arrays1)
# split_arrays2 = np.split(array_1d,[1,3],axis=1)
# print_splits(split_arrays2)
"""
# Split-Horizontal:-
# "np.hsplit(arr,indices_or_sections)".
# --> Split the given 'arr' into multiple sub-arrays horizontally i.e column-wise and returns a list of sub-arrays.
# --> It is equivalent to split with 'axis = 1'.
"""
array = np.array([1,2,3,4,5])
split_arrays = np.hsplit(array,[3])
print_splits(split_arrays)

array = np.array([[1,-1,2,-2,3,-3],
                  [1,-1,2,-2,3,-3],
                  [1,-1,2,-2,3,-3]])
split_arrays = np.hsplit(array,[2])
print("split arrays:")
print_splits(split_arrays)
split_arrays = np.hsplit(array,[1,2])
print("split arrays:")
print_splits(split_arrays)
"""
# Split-Vertical:-
# "np.vsplit(arr,indices_or_sections)".
# --> Splits yhe given 'arr' into multiple sub-arrays vertically i.e row-wise and returns a list of sub-arrays.
# --> it is equivalent to split with 'axis = 0'.
# --> '.vsplit' only works on arrays of 2 or more dimensions.
"""
array = np.array([[1,1,1],
                  [-1,-1,-1],
                  [2,2,2],
                  [-2,-2,-2],
                  [3,3,3],
                  [-3,-3,-3]])
split_arrays = np.vsplit(array,[2])
print("split arrays:")
print_splits(split_arrays)
split_arrays = np.vsplit(array,[2,3])
print_splits(split_arrays)
"""
# ---------------------------------------------------------

# Joining Arrays:-

# Concatenation of arrays:-
# "np.concatenate((a1,a2,a3,......),axis = 0)".
# --> Joins a sequence of arrays along the given axis and returns the concatenated array.
# --> The arrays must have the same shape, except in the dimension corresponding to the 'axis'.
# --> The resulting array has the same dimensions as that of the input array.
"""
a = np.array([7,1,11])
b = np.array([3,0,17])
print(np.concatenate((a,b)))
"""
# Concatenate along 'axis = 0'.
"""
a = np.array([[7,4,5],
              [1,3,2]])
b = np.array([[-1,-2,-3]])
print(np.concatenate((a,b),axis=0))
"""
# Concatenate along 'axis = 1'.
"""
a = np.array([[7,4,5],
              [1,3,2]])
b = np.array([[-1],
              [-2]])
print(np.concatenate((a,b),axis=1))
"""
# The split arrays obtained by "np.split()" can be concatenated using "np.concatenate()" to get the original array.
"""
array = np.array([[1,1,1],
                  [-1,-1,-1],
                  [2,2,2],
                  [-2,-2,-2],
                  [3,3,3],
                  [-3,-3,-3]])
split_arrays = np.split(array,[2],axis=0)
print("split arrays:")
print_splits(split_arrays)
print(np.concatenate(split_arrays,axis=0))
"""
# ---------------------------------------------------------

# Stacking - Vertical:-

# "np.vstack((a1,a2,..........))".
# --> Stacks the arrays "a1,a2,..." vertically (row-wise) in sequence and returns the stacked array.
# --> Except in the case of 1D arrays, it's equivalent to "np.concatenate" along "axis = 0".
# 1-D arrays must have the same length to apply "np.vstack" on them.
"""
a = np.array([1,7,11]) # 1D array
b = np.array([10,20,30])
vstack_array = np.vstack((a,b))
print(vstack_array)
"""
# nD-arrays(n>1) must have the same shape along every axis except for the first axis (axis-0).
"""
a = np.array([[1,2,3],
              [4,5,6]])
b = np.array([[-1,-2,-3],
              [-4,-5,-6],
              [-7,-8,-9],
              [-10,-11,-12]])
print(np.vstack((a,b)))
"""
# ---------------------------------------------------------

# Stacking - Horizontal:-

# "np.hstack((a1,a2,...))".
# --> Stacks the arrays "a1,a2,..." horizontally (column-wise) in sequence and returns the stacked array.
# --> Except in the case of 1D arrays, its equivalent to "np.concatenate" along "axis = 1".
# 1-D arrays can be of any length.
"""
a = np.array([1,7,11])
b = np.array([10,20])
print(np.hstack((a,b)))
"""
# nD-arrays(n>1) must have the same shape along every axis except for the second axis (axis-1).
"""
a = np.array([[1,2],
              [3,4],
              [5,6]])
b = np.array([[-1,-2,-3,-4],
              [-5,-6,-7,-8],
              [-9,-10,-11,-12]])
print(np.hstack((a,b)))
"""
# ---------------------------------------------------------

# Stacking:-

# "np.stack(arrays,axis = 0)".
# --> Joins a sequence of arrays along the new axis.
# --> All input arrays must have the same shape.
# --> The resulting array has "1" additional dimension to  the input arrays.
# --> The "axis" parameter specifies the index of the new axis which has to be created.
"""
a = np.array([1,2,3])
b = np.array([-1,-2,-3])

stacked_axis_0 = np.stack([a,b],axis=0)
print(stacked_axis_0)
print(stacked_axis_0.shape)

stacked_axis_1 = np.stack([a,b],axis=1)
print(stacked_axis_1)
print(stacked_axis_1.shape)
"""
"""
a = np.array([[1,2,3],
              [4,5,6]])
b = np.array([[-1,-2,-3],
              [-4,-5,-6]])
stacked_axis_2 = np.stack([a,b],axis=2)
print("Stacked array with axis-2:\n",stacked_axis_2)
print("Shape of the given arrays:",a.shape,b.shape)
print("Shape of the stacked arrays:",stacked_axis_2.shape)
"""
# The size of the new dimension which is created will be equal to the number of arrays that are stacked.
"""
arrays = [np.random.randn(2,3) for i in range(5)]
print(np.stack(arrays,axis=2))
print(np.stack(arrays,axis=2).shape)
"""
# ---------------------------------------------------------

# Tiling Arrays:-

# np.repeat():-
# "numpy.repeat(arr,num_repeats,axis=None)".
# --> Repeats elements of an array.
# --> Outputs an array which has the same shape as 'arr', except along the given 'axis'.
# --> 'num_repeats' are the number of repitions for each element which is broadcasted to fit the shape of the given 'axis'.
"""
print(np.repeat(2,5))

a = np.array([[1,2,3],
              [4,5,6]])
print(np.repeat(a,2,axis=0))

a = np.array([[1,2,3],
              [4,5,6]])
print(np.repeat(a,3,axis=1))
"""
# 'num_repeats' can also be a list that indicates how many times each of the corresponding elements should be repeatd (along the given axis).
"""
a = np.array([[1,2,3],
              [4,5,6]])
print(np.repeat(a,[3,5],axis=0))
"""
# ---------------------------------------------------------

# Adding and removing elements:-

# np.delete():-
# "np.delete(arr,delete_indices,axis = None)".
# --> Returns a new array by deleting all the sub-arrays along the mentioned 'axis'.
# 'delete_indices' indicates the indices of the sub-arrays that need to be removed along the specified 'axis'.
"""
a = np.array([[1,3,5,7],
              [2,4,6,8],
              [-1,-2,-3,-4]])
delete_in_axis_0 = np.delete(a,1,0)
print("Delete at position-1 along axis-0:\n",delete_in_axis_0)

a = np.array([[1,3,5,7],
              [2,4,6,8],
              [-1,-2,-3,-4]])
delete_in_axis_1 = np.delete(a,2,1)
print("Delete at position-2 along axis-1:\n",delete_in_axis_1)

a = np.array([[1,3,5,7],
              [2,4,6,8],
              [-1,-2,-3,-4]])
print(np.delete(a,[0,5],axis=None))
"""

# np.insert():-
# "numpy.insert(arr,indices,values,axis=None)".
# --> Insert values along the given axis before the given indices.
# --> 'indices' defines the index (or) indices before which the given 'values' are inserted.
# --> 'values' to insert into 'arr'. If the type of values is different from that of 'arr', 'values' is converted to the type of 'arr'. Values should be shaped so that 'arr[...,indices,...] = values' is legal.
# --> Output is a copy of 'arr' with values appended to the specified 'axis'.
"""
a = np.array([[1,-1],
              [2,-2],
              [3,-3]])
insert_axis_0 = np.insert(a,1,[4,4],axis=0)
print("Insert 4's at position-1 along axis-0:\n",insert_axis_0)
insert_axis_1 = np.insert(a,1,4,axis=1)
print("Insert 4's at position-1 along axis-1:\n",insert_axis_1)
"""
"""
a = np.array([[1,-1],
              [2,-2],
              [3,-3]])
zeros = np.zeros((3,2))
print(np.insert(a,[1],zeros,axis=1))

a = np.array([[1,-1],
              [2,-2],
              [3,-3]])
print(np.insert(a,1,4,axis=None))
"""

# np.append():-

# "numpy.append(arr,values,axis=None)".
# --> Append values to the end of an array.
# --> Output is a copy of 'arr' with values appended to 'axis'.
# If axis is given, both 'arr' and 'values' should have the same shape.
"""
array_2d = np.array([[1,2,3],
                     [4,5,6]])
values = np.array([[-1,-2,-3],
                   [-4,-5,-6]])
print(np.append(array_2d,values,axis=0))
print(np.append(array_2d,values,axis=1))
"""
# If 'axis' is not given, both 'arr' and 'values' are flattened before use.
"""
print(np.append(array_2d,values))
"""
# ---------------------------------------------------------

# Padding:-

# "numpy.pad(array,pad_width,mode = 'constant',**kwargs)".
# --> 'pad_width' is the number of values padded to the edges of each 'axis'.
# --> Output is a padded array of rank equal to 'array', with shape increased according to 'pad_width'.
# "mode = 'constant' (default)".
# --> Pads with a constant value.
# "constant_values": The padded values to set for each axis.
"""
array = np.array([[1,3],[7,9]])
pad_array_constant = np.pad(array,(1,2),'constant',constant_values=(-6,-5))
print("pad_array_constant:\n",pad_array_constant)
"""
# "mode = 'edge'".
# --> Pads with the edge values of the array.
"""
array = np.array([[1,3],[7,9]])
pad_array_edge = np.pad(array,(1,1),'edge')
print("pad_array_edge:\n",pad_array_edge)
"""
# ---------------------------------------------------------

# Understanding NumPy Internals:-

original_3D = np.array([[['A','B'],
                         ['C','D'],
                         ['E','F']],
                        [['G','H'],
                         ['I','J'],
                         ['K','L']],
                        [['M','N'],
                         ['O','P'],
                         ['Q','R']],
                        [['S','T'],
                         ['U','V'],
                         ['W','X']]])
"""
print("original_3d shape:",original_3D.shape)
print("original_3d strides:",original_3D.strides)
print("original_3d data:",original_3D.data)
"""
print(original_3D)

# np.ravel:-
"""
ravel_3d = original_3D.ravel()
print("ravel_3d shape:",ravel_3d.shape)
print("ravel_3d strides:",ravel_3d.strides)
print(ravel_3d.base is original_3D)
"""
# np.swapaxes:-
"""
swap_3d = np.swapaxes(original_3D,1,2)
print("swap_3d:",swap_3d)
print("swap_3d shape:",swap_3d.shape)
print("swap_3d strides:",swap_3d.strides)
print(swap_3d.base is original_3D)
"""
# np.moveaxis:-
"""
move_3d = np.moveaxis(original_3D,[1],[2])
print("move_3d:\n",move_3d)
print("move_3d shape:",move_3d.shape)
print("move_3d strides:",move_3d.strides)
print(move_3d.base is original_3D)
print(move_3d[0,0,0])
print(move_3d[0,0,1])
"""
# np.reshape:-
"""
reshape_3d = np.reshape(original_3D,(4,2,3))
print("reshape_3d:\n",reshape_3d)
print("reshape_3d shape:",reshape_3d.shape)
print("reshape_3d strides:",reshape_3d.strides)
print(reshape_3d.base is original_3D)
print(reshape_3d[0,0,0])
print(reshape_3d[0,0,1])
"""
# Slicing:-
"""
sliced_3d = original_3D[::2,:,::2]
print("sliced_3d:\n",sliced_3d)
print("sliced_3d shape:",sliced_3d.shape)
print("sliced_3d strides:",sliced_3d.strides)
print(sliced_3d.base is original_3D)
"""
# np.transpose:-
"""
transpose_3d = original_3D.T
print("transposed_3d:",transpose_3d)
print("transposed_3d shape:",transpose_3d.shape)
print("transposed_3d strides:",transpose_3d.strides)
print(transpose_3d.base is original_3D)
"""