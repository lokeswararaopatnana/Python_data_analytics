# importing numpy
import numpy as np

# Operations on Numpy Arrays:-

# Addinf Numpy Arrays vs Python Lists:-

# In Python, adding two Python Lists will create a new list that contains elements from both the lists in the given order (List Concatenation).
"""
concatenated_list = [1,5,1,0,7] + [2,4,3,1,3]
print("Concatenated List:",concatenated_list)

# But, when we add two NumPy Arrays, elements of one array will be added to the corresponding elements in the other array (Vectorized Addition).

sum_of_two_arrays = np.array([1,5,13,0,7]) + np.array([2,4,8,7,13])
print("Sum of two arrays:",sum_of_two_arrays)

"""
# ---------------------------------------------------------

# Operator Overloading:-
"""
a = np.array([1,5,1,0,7])
b = np.array([2,4,3,1,3])

sum_of_a_b = np.add(a,b)
print("Sum of two arrays with add function:",sum_of_a_b)

sum_with_operator = a + b
print("Sum of two arrays with + operator:",sum_with_operator)

"""
# ---------------------------------------------------------

# Vectorized Operations with NumPy:-
"""
vec_size = 1000
a = np.arange(vec_size)
b = np.arange(vec_size)

def addition_with_loop():
    sum_a_b = [a[i] + b[i] for i in range(len(a))]
    return sum_a_b

def vectorized_np_addition():
    sum_a_b = a + b
    return sum_a_b

print(addition_with_loop())
print(vectorized_np_addition())

"""
# As we have seen with addition, many other operators also support Vectorized Operations (We can apply element-wise operations without having to iterate over all the elements individually).
# This enables us to do a lot of computations much faster and with less code.

# -----------------------------------------------------------

# Array Operations:-

# Arithmetic Operations:-
"""
  |Arithmetic Operation | Numpy Function | Python Operator|
  |-------------------  | -------------- | ---------------|
  |     Addition        |    np.add      |        +       |
  |     Subtraction     |   np.subtract  |        -       |
  |     Multiplication  |   np.multiply  |        *       |
  |     Division        |   np.divide    |        /       |
  |   Integer Division  | np.floor_divide|       //       |
  |   Modulo Operation  |    np.mod      |       %        |
  |      Power          |    np.power    |       **       |
  ---------------------------------------------------------
"""
"""
a = np.array([1,5,1,0,7])
b = np.array([2,4,3,1,3])

difference = np.subtract(a,b)
print("Difference:",difference)

product = np.multiply(a,b)
print("Product:",product)

division = np.divide(a,b)
print("Division:",division)

floor_division = np.floor_divide(a,b)
print("Floor_division:",floor_division)

remainder = np.mod(a,b)
print("Remainder:",remainder)

a_power_b_with_function = np.power(a,b)
print("a_power_b_with_function:",a_power_b_with_function)

a_power_b_with_operator = a**b
print("a_power_b_with_operator:",a_power_b_with_operator)

"""
# Relational Operations:-
"""
  |Relational Operation | Numpy Function | Python Operator|
  |-------------------  | -------------- | ---------------|
  |    Greater Than     |    np.greater  |        >       |
  | Greater Than Equal  |np.greater_equal|        >=      |
  |      Less Than      |   np.less      |        <       |
  |   Less Than Equal   |  np.less_equal |        <=      |
  |        Equal        |    np.equal    |        ==      |
  |      Not Equal      |  np.not_equal  |        !=      |
  ---------------------------------------------------------
"""
# Relational Opeartions return Boolean Arrays
"""
a = np.array([1,5,1,0,7])
b = np.array([2,4,3,1,3])

print("np.greater(a,b):",np.greater(a,b))
print("          a > b:",a > b)

print("np.greater_equal(a,b):",np.greater_equal(a,b))
print("               a >= b:",a >= b)

print("np.less(a,b):",np.less(a,b))
print("       a < b:",a < b)

print("np.less_equal(a,b):",np.less_equal(a,b))
print("            a <= b:",a <= b)

print("np.equal(a,b):",np.equal(a,b))
print("       a == b:",a == b)

print("np.not_equal(a,b):",np.not_equal(a,b))
print("           a != b:",a != b)

"""
# Bitwise Operations:-
"""
  |Relational Operation | Numpy Function | Python Operator|
  |-------------------  | -------------- | ---------------|
  |     Bitwise AND     | np.bitwise_and |        &       |
  |     Bitwise OR      | np.bitwise_or  |        |       |
  |     Bitwise XOR     | np.bitwise_xor |        ^       |
  |     Bitwise NOT     |   np.invert    |        ~       |
  |      Left Shift     | np.left_shift  |        <<      |
  |     Right Shift     | np.right_shift |        >>      |
  ---------------------------------------------------------
"""
"""
a = np.array([1,0,1,0,1])
b = np.array([1,0,0,1,1])

print("np.bitwise_and(a,b):",np.bitwise_and(a,b))
print("              a & b:",a & b)

print("np.bitwise_or(a,b):",np.bitwise_or(a,b))
print("             a | b:",a | b)

print("np.bitwise_xor(a,b):",np.bitwise_xor(a,b))
print("              a ^ b:",a ^ b)

print("np.invert(a):",np.invert(a))
print("         ~ a:", ~ a)

print("np.left_shift(a,b):",np.left_shift(a,b))
print("            a << b:",a << b)

print("np.right_shift(a,b):",np.right_shift(a,b))
print("             a >> b:",a >> b)

"""
# ---------------------------------------------------------

# Operations on 2D Arrays:-
# All the arithmetic, realtional and bitwise operations can be done on 2D, 3D arrays and in general on nD arrays as well.
"""
a = np.array([[1,3,2],
              [0,2,-1]])
b = np.array([[1,1,1],
              [2,2,2]])
print("a + b:\n",a + b)
print("a > b:\n",a > b)

"""
# ---------------------------------------------------------
# Linear Algebra:-

# "numpy.matmul":-
# The "matmul" function implements the semantics of the "@" operator introduced in Python-3.5.
"""
a = np.array([[3,1,2],
              [4,2,1]])
b = np.array([[2,3],
              [-1,0],
              [-3,4]])

print("Shape of a:",a.shape)
print("Shape of b:",b.shape)

print("np.matmul(a,b):\n",np.matmul(a,b))
print("         a @ b:\n",a @ b)

"""
# "matmul" in the case of Rank 1 Arrays:-
"""
a = np.array([3,1,2,4])
b = np.array([5,2,0,1])

print("Shape of a:",a.shape)
print("Shape of b:",b.shape)

print("n.matmul(a,b):\n",np.matmul(a,b))
print("        a @ b:\n",a @ b)

"""
# Reshaping Rank-1 Arrays:-
"""
a = np.array([3,1,2,4])
b = np.array([5,2,0,1])

print("Shape of a:",a.shape)
print("Shape of b:",b.shape)

a = a.reshape(1,4)
b = b.reshape(4,1)

print("np.matmul(a,b):\n",np.matmul(a,b))
print("         a @ b:\n",a @ b)

print("np.matmul(b,a):\n",np.matmul(b,a))
print("         b @ a:\n",b @ a)

"""
# Transpose of an array:-
"""
a = np.array([[3,1,2],
              [4,2,1]])
transposed_array = np.transpose(a)
print(transposed_array)

# Can also use "a.T" to transpose an array.
a = np.array([[3,1,2],
              [6,4,5]])
transposed_array = a.T
print(transposed_array)

"""
# Transposing Rank 1 arrays:-
"""
a = np.array([1,2,3,4]).reshape(1,4)
print("Array before transposing:\n",a)

b = np.transpose(a)
print("Array after transposing:\n",b)

a = np.array([1,2,3,4])
print("Array before transposing:\n",a)

b = np.transpose(a)
print("Array after transposing:\n",b)

"""
# ---------------------------------------------------------
# Other linear algebra functions:-
"""
# Determinant of an array.
a = np.array([[1,2],
              [1,3]])

detreminant = np.linalg.det(a)
print("Determinant of array:",detreminant)

# Multiplicative inverse of a matrix.
a = np.array([[1,2],
              [1,3]])
inverse = np.linalg.inv(a)
print("Inverse of array:\n",inverse)

"""
# ---------------------------------------------------------
# Broadcasting:-
"""
Broadcasting two arrays together follows these rules:
  # Two arrays are said to be compatible in a dimension if
    --> they have the same size in that dimension,(or).
    --> one of the arrays has size '1' in that dimension.
  # The arrays can be broadcasr together if they are compatible in all dimensions.
  # After broadcasting, each array behaves as if its shape is equal to the element-wise maximum of the shapes of two input arrays.
  # In any dimension where one array has a size of '1' and the other array has a size greater than 1, the first array behaves as if it were copied along that dimension.
"""
"""
a = np.zeros((3,4))
b = np.array([1,2,3,-1])

print("a:\n",a)
print("b:\n",b)
print("a + b:\n",a + b)

a = np.zeros((3,4))
b = np.array([[1],
              [2],
              [3]])
print("a:\n",a)
print("b:\n",b)
print("a + b:\n",a + b)

a = np.zeros((3,4))
print("a:\n",a)
print("a + 5:\n",a + 5)

a = np.array([[3,1,2,-1],
              [4,2,1,0],
              [-2,1,3,4]])
b = np.array([1,0,2,1])
print("a:\n",a)
print("b:\n",b)
print("a + b:\n",a + b)

"""
# Broadcasting and Masking:-
"""
A = np.array([[3,1,4,2],
              [0,-4,7,9]])
mask = A > 0
print("mask created with broadcasting:\n",mask)
print("Applying mask on A:\n",A[mask])

"""
# ---------------------------------------------------------
# Other Useful Methods in NumPy:-
"""
# A universal function (or ufunc for short) is a function that operates on ndarrays in an element-by-element fashion, supporting array broadcasting, type casting, and several other standard features.
# There are currently more than 60 universal functions defined in numpy on one or more types, covering a wide variety of operations. 
"""
# Sum of the elements of array:-
# "np.sum(a, axis = None)".
# ndarray.sum(axis = None).
# Returns the sum of array elements along the given axis. If axis is None, then it computes the sum of all the elements in the array.
"""
array_1d = np.array([1,2,3])
print("Sum of all elements:",np.sum(array_1d))
print("Sum of all elements alternative:",array_1d.sum())

array_2d = np.array([[1,2,1],
                     [-1,0,3]])
array_2d_sum = array_2d.sum()
print("Sum of all elements:",array_2d_sum)

# Sum along different axes:-

sum_axis_0 = array_2d.sum(axis=0)
print("array_2d:\n",array_2d)
print("Sum along axis 0:\n",sum_axis_0)

sum_axis_1 = array_2d.sum(axis=1)
print("array_2d:\n",array_2d)
print("Sum along axis 1:",sum_axis_1)

"""
# Sum in 3D arrays:-
"""
array_3d = np.array([[[1,0],
                      [2,3],
                      [1,4],
                      [-1,2]],
                     [[-1,0],
                      [1,4],
                      [1,2],
                      [3,-1]],
                     [[-4,1],
                      [2,0],
                      [1,2],
                      [0,1]]])
print("array_3d:\n",array_3d)
print("Sum of all elements in array_3d:",array_3d.sum())
print("Sum along axis 0:\n",array_3d.sum(axis=0))
print("Sum along axis 1:\n",array_3d.sum(axis=1))
print("Sum along axis 2:\n",array_3d.sum(axis=2))

"""
# ---------------------------------------------------------
"""
# "np.add.reduce()":-

# "np.add.reduce()" is equivalent to "np.sum()".
# "np.sum()" internally calls "np.add.reduce".
array_2d = np.array([[1,2,1],
                     [-1,0,3]])
sum_1 = np.sum(array_2d,axis=0)
print("using np.sum:",sum_1)

sum_2 = np.add.reduce(array_2d,axis=0)
print("using np.add.reduce:",sum_2)

# Maximum of the elements in an array:-

array_2d = np.array([[1,2,1],
                     [-1,0,3]])

max_of_array = array_2d.max()
print("max of array:",max_of_array)

max_of_array = np.max(array_2d)
print("max of array alternative:",max_of_array)

max_axis_0 = array_2d.max(axis=0)
print("max along axis 0:",max_axis_0)

max_axis_1= array_2d.max(axis=1)
print("max along axis 1:",max_axis_1)

array_2d = np.array([[1,2,1],
                     [-1,0,3]])
sum_2 = np.maximum.reduce(array_2d,axis=0)
print("using np.maximum.reduce:",sum_2)
sum_2 = np.maximum.reduce(array_2d,axis=1)
print("using np.maximum.reduce:",sum_2)

# Argmax:-

array_1d = np.array([1,-1,3,0,2])
print("maximum element:",np.max(array_1d))
print("maximum element is at index:",np.argmax(array_1d))

"""
# ---------------------------------------------------------
# Sorting:-

# In-place sorting vs New sorted array:-
# "np.sort()" returns a sorted copy of an array.
"""
array_1d = np.array([1,-1,3,0,2])
sorted_array = np.sort(array_1d)
print("array_1d:",array_1d)
print("Sorted array:",sorted_array)

# "ndarray.sort()" sorts an array in-place.

array_1d = np.array([1,-1,0,3,2])
array_1d.sort()
print("array_1d:",array_1d)

# "np.argsort()":-
# returns the indices that would sort an array.
array_1d = np.array([4,2,0,1,5])
indices = np.argsort(array_1d)
print("indices that would sort array_1d:",indices)
print("Using fancy indexing with argsort:",array_1d[indices])

"""