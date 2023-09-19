import numpy as np

# NumPy Arrays Vs Python Lists

# By adding two simple vectors, you can obsrve the time taken by using python lists vs using NumPy arrays.

"""
vec_size = 1000
def add_python_lists():
    x = range(vec_size)
    y = range(vec_size)
    z = [x[i] + y[i] for i in range(len(x))]
    return z

def add_numpy_arrays():
    a = np.arange(vec_size)
    b = np.arange(vec_size)
    c = a + b
    return c

"""
# "%timeit" function: 'N' loops indicate that the function has been called 'N' times to compute the average time to excute the function.
# Best of 'K' indicates that the "%timeit" has been run 'K' time to take the best average.
# These constants can also be modified and given as arguments to the "%timeit" function. To know more, run "%timeit?" in a new cell.
"""

print(add_python_lists())
print(add_numpy_arrays())

"""
# ---------------------------------------------------------

# Creating NumPy Arrays from Lists

# Numpy arrays can be created using a list with "numpy.array".
# All elements of a numpy array should be of the same type.
    # "numpy.array" chooses a type which can fit all of the elements of the array.
    # We can also specify type of elements with "dtype" parameter.

""" 
np_array_from_list = np.array([2,3,5,7,11])
print("np_array_from_list:",np_array_from_list)
print(np_array_from_list.dtype)

bool_array = np.array([False, True, True, False])
print("bool_array:",bool_array)
print(bool_array.dtype)

type_casted_float_array = np.array([0,1,2,3,4,5,6,7,8,9], dtype=float)
print("type_casted_float_array:",type_casted_float_array)
print(type_casted_float_array.dtype)

int_array = np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.int16)
print("type_casted_float_array:",int_array)
print(int_array.dtype)

type_casted_int_array = np.array([0,1,2,3,4.666,5,6,7.2544,8,9.2154], dtype=np.int64)
print("type_casted_float_array:",type_casted_int_array)
print(type_casted_int_array.dtype)

""" 
# ---------------------------------------------------------

# dtype-Precision

"""
print(np.array([1.00001,1.00001],dtype=np.float32))

print(np.array([1.00000021,1.00000021],dtype=np.float32))

print(np.array([1.00000021,1.00000021],dtype=np.float64))

"""
# ---------------------------------------------------------

# Creating 2D Nummpy Arrays

"""
array_2d = np.array([[1,2,3],
                     [4,5,6]])
print(array_2d)

# "ndarray" Properities
# To return the number of elements in an array, we can use the size attribute, as shown below:
print(array_2d.size)

# The number of dimensions of array can be obtained via the "ndim" attribute:
print(array_2d.ndim)

# The number of elements along each array dimension can be obtained via "shape" attribute:
print(array_2d.shape)

# By default, NumPy infers the type of the array upon construction. Since we passed integers to the array, the "ndarray" object 'array_2d' should be of the type "int64" on a 64-bit machine, which we can confirm by accessing the "dtype" attribute.
print(array_2d.dtype)

"""
# ---------------------------------------------------------

# Creating NumPy Arrays with Constant Values

# "np.ones"
    # np.ones(shape, dtype=float)
        # Returns a new array of given shape and type, filled with ones.

"""
ones_5_1d = np.ones(5)
print("ones_5_1d:",ones_5_1d)
print(ones_5_1d.dtype)

ones_2d = np.ones([3,4])
print("ones_2d:",ones_2d)
print(ones_2d.dtype)

ones_2d_int_type = np.ones([2,3], dtype=np.int32)
print("ones_2d_int_type:",ones_2d_int_type)
print(ones_2d_int_type.dtype)

"""

# "np.zeros"
    # np.zeros(shape, dtype=float)
        # Returns a new array of given shape and type, filled with zeros.

"""
zeros_5 = np.zeros(5)
print("zeros_5:",zeros_5)
print(zeros_5.dtype)

zeros_int_2d = np.zeros([3,5], dtype=np.int32)
print("zeros_int_2d:",zeros_int_2d)
print(zeros_int_2d.dtype)

"""

# "np.full"
    # np.full(shape, fill_value, dtype=None)
        # Returns a new array of the given shape and type, fiilled with 'fill_value'.
        # The default data type is derived from the 'fill_value'.
        
"""
full_array = np.full([2,2], fill_value=14.3)
print("full_array:\n",full_array)
print(full_array.dtype)

full_int_array = np.full([3,4], fill_value=6.8, dtype=int)
print("full_int_array:\n",full_int_array)
print(full_int_array.dtype)

full_bool_array = np.full([4,3],fill_value=False)
print("full_bool_array:",full_bool_array)
print(full_bool_array.dtype)

""" 
# ---------------------------------------------------------

# Other Array Construction Methods

# "np.empty()"

"""
empty_array = np.empty(3, dtype=int)
print("empty_array:",empty_array)
print(empty_array.dtype)

empty_array_2d = np.empty([2,3], dtype=np.float32)
print("empty_array_2d:",empty_array_2d)
print(empty_array_2d.dtype)

"""

# Identity matrix
"""
identity_matrix_float = np.eye(5)
print(identity_matrix_float)
print(identity_matrix_float.dtype)

identity_matrix_int = np.eye(6,dtype=int)
print(identity_matrix_int)
print(identity_matrix_int.dtype)

identity_matrix_2d_float = np.eye(2,3,)
print(identity_matrix_2d_float)

identity_matrix_2d_int = np.eye(2,3,dtype=int)
print(identity_matrix_2d_int)

"""

# Diagonal Matrix
"""
diagonal_matrix_1d = np.diag((1,2,3,4,5))
print(diagonal_matrix_1d)

"""
# ---------------------------------------------------------

# Creating NumPy Sequential Arrays

# "np.arange"
    # np.arange(start,stop,step,dtype=None)
        # Equivalent to the Python built-in range, but returns an 'ndarray' rather than a list.
        # Values are generated within the half-open interval [start,stop)
        # Much more efficient than 'np.array(range(_))'.
"""
numbers_array = np.arange(10)
print("numbers_array_first_10_numbers:",numbers_array)

numbers_list = list(range(10))
print("numbers_list_first_10_numbers:",numbers_list)

numbers_with_start_and_stop = np.arange(start=2,stop=11)
print("numbers_with_start_and_stop:",numbers_with_start_and_stop)

numbers_with_step = np.arange(start=3,stop=15,step=2)
print("numbers_with_step:",numbers_with_step)

numbers_with_neg_step = np.arange(15,-1,-2)
print("numbers_with_neg_step:",numbers_with_neg_step)

numbers_with_float_step = np.arange(start=1,stop=2,step=0.11111111)
print("numbers_with_float_step:",numbers_with_float_step)

"""

# "np.linspace"
    # np.linspace(start,stop,num=50,endpont=True,retstep=False,dtype=None)
        # Returns evenly spaced numbers, calculated over the interval [start,stop]. The number of evenly spaced numbers returned is equal to 'num'.
        # Can control whether the last number 'stop' can be included or excluded with the 'endpoint' parameter.
        
"""
seq_array = np.linspace(start=2.0,stop=4.0,num=5)
print("sequential array with linear spacing:",seq_array)

seq_array_without_num = np.linspace(start=3.0,stop=5.0) # when we don't give 'num' value here it's automatically gives 50 numbers in array.
print("linspace array without num:",seq_array_without_num)

seq_array_excluding_endpoint = np.linspace(start=7.0,stop=9.0,num=6,endpoint=False)
print("linspace array excluding endpoint:",seq_array_excluding_endpoint)

seq_array_with_retstep = np.linspace(start=9.0,stop=10.0,endpoint=False,retstep=True)
print("linspace array with retstep:",seq_array_with_retstep)
print(seq_array_with_retstep[0])

seq_2d_array = np.linspace(start=[2,3,5],stop=[4,6,7],num=5)
print("linspace array in 2d: \n",seq_2d_array)

seq_2d_array_axis_1 = np.linspace(start=[2,3,5],stop=[4,6,7],num=5,axis=1)
print("linspace array in 2d along axis 1: \n",seq_2d_array_axis_1)

geo_space = np.geomspace(start=1,stop=128,num=8)
print("geometrically spaced array:",geo_space)

log_space = np.logspace(start=2,stop=10,num=5,base=2)
print("geometrically spaced array:",log_space)
"""

# ---------------------------------------------------------

# Visualisation

# Visualising the numpy arrays created using 'np.linspace'.
"""
import matplotlib.pyplot as plt 
N1 = 6
N2 = 6
x1 = np.linspace(0,10,N1,endpoint=True)
x2 = np.linspace(0,10,N2,endpoint=False)
y1 = np.zeros(N1)
y2 = np.zeros(N2) + 0.2
plt.plot(x1,y1,'o')
plt.plot(x2,y2,'o')
plt.ylim([-0.5,1])
plt.show()

"""

# "np.random.random()"
    # 'np.random.random' creates an array of the given shape with random values in the interval [0,1].
    
"""
random_value_array = np.random.random(6)
print("random_valued_array:",random_value_array)

random_value_array_2d = np.random.random([3,2])
print("random_valued_array_2d:",random_value_array_2d)

"""

# "nD Arrays"

"""
array_1d = np.array(['A','B','C'])
print(array_1d,array_1d.dtype)

array_2d = np.array([['A','B','C'],
                     ['D','E','F'],
                     ['G','H','I']])
print(array_2d,array_2d.dtype)

array_3d = np.array([[['A','B','C'],
                      ['D','E','F'],
                      ['G','H','I']],
                     [['J','K','L'],
                      ['M','N','O'],
                      ['P','Q','R']],
                     [['S','T','U'],
                      ['V','W','X'],
                      ['Y','Z','A']]])
print(array_3d,array_3d.dtype)

array_4d = np.array([[[['A','B'],
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
               ['W','X']],
              [['Y','Z'],
               ['A','B'],
               ['C','D']],
              [['E','F'],
               ['G','H'],
               ['I','J']]]])
print(array_4d,array_4d.dtype)

ones_3d = np.ones([2,3,4],dtype=np.int32)
print(ones_3d)

zeros_4d = np.zeros([2,3,3,2],dtype=np.int32)
print(zeros_4d)

"""

# Summary: Creating NumPy Arrays
    # np.array
    # np.zeros,np.ones,np.full
    # np.arange,np.linspace
    # np.empty
    # np.random.random
    # nD Arrays