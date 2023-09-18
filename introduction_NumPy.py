import numpy as np

# Numpy Arrays vs Python Lists
""" By adding two simple vectors, you can observe the time taken by using python lists vs using NumPy arrays. """

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

# %timeit function: N loops indicate that the function has been called N times to compute the average time to execute the function.

# Best of K indicates that the %timeit has been run K times to take the best average.

# These constants can also be modified and given as arguments to the %timeit function. To know more, run %timeit? in a new cell.

print(add_python_lists())
print(add_numpy_arrays())

"""

# ---------------------------------------------------------
""" Creating NumPy Arrays """
""" Creating NumPy Arrays: From Lists """
# Numpy arrays can be created using a list with numpy.array
# All elements of a numpy array should be of the same type
    # numpy.array chooses a data type which fits all of the elements in the array
    # We can also specify the data type with the dtype parameter

"""

np_array_list = np.array([1,3,5,7,9])
print(np_array_list)

print("------------------------------")

boolean_array_list = np.array([True,False,False,True])
print(boolean_array_list)
print("------------------------------")

typecasted_array_list = np.array([1,2,3,4,5,6],dtype=float)
print(typecasted_array_list)

print("------------------------------")

int_array_16 = np.array([2,4,6,8,10],dtype=np.int16)
print(int_array_16)

print("------------------------------")

typecasted_int_array_32 = np.array([2,4,6,8,10],dtype=np.int32)
print(typecasted_int_array_32)

print("------------------------------")

typecasted_int_array_64 = np.array([2,4,6,8,10],dtype=np.int64)
print(typecasted_int_array_64)

"""
# ---------------------------------------------------------

# Creating NumPy Constant Arrays

# "np.zeros"

# np.zeros(shape,dtype=float)
    # Returns a new array of given shape and type, filled with zeros.

"""
zeros_5 = np.zeros(5)
print("zeros_5:",zeros_5)

print("------------------------------------")

zeros_7 = np.zeros(7,dtype=int)
print("zeros_7:",zeros_7)

"""

# "np.ones"

# np.ones(shape,dtype=float)
    # Returns a new array of given shape and type, filled with ones.

""" 
ones_5 = np.ones(5)
print("ones_5:",ones_5)

print("-------------------------------------")

ones_7 = np.ones(7,dtype=int)
print("ones_7:",ones_7)

""" 

# "np.full"

# np.full(shape, fill_value, dtype=None)
    # Returns a new array of given shape and type, filled with fill_value.
    # The default data type is derived from the "fill_value".
    
"""
fill_array = np.full(10,fill_value=5.7)
print("fill_array:",fill_array)

print("--------------------------------------------------")

fill_int_array = np.full(10,fill_value=5.7,dtype=int)
print("fill_int_array:",fill_int_array)

print("--------------------------------------------------")

fill_bool_array = np.full(10,fill_value=True)
print("fill_bool_array:",fill_bool_array)

"""

# ---------------------------------------------------------

# Creating NumPy Sequential Arrays

# "np.arange"

# np.arange(start,stop,step,dtype=None)
    # Equivalent to the Python built-in range, but returns an ndarray rather than a list.
    # Values are generated within the half-open interval [start,stop).
    # Much more efficient than np.array(range(_)).

"""
first_10_numbers = np.arange(10)
print("first_10_numbers:",first_10_numbers)

print("--------------------------------------------------")

number_with_step = np.arange(5,20,2,dtype=float)
print("number_with_step:",number_with_step)

print("--------------------------------------------------")

number_with_neg_step = np.arange(10,0,-1)
print("number_with_neg_step:",number_with_neg_step)

"""

# "np.linspace"

# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    # Returns evenly spaced numbers, calculated over the interval [start,stop]. The number of evenly spaced numbers returned is equal to "num".
    # Can control whether the last number "stop" can be included or excluded with the "endpoint" parameter.

"""
a = 5.0
b = 7.0
n = 10

sequential_array = np.linspace(a,b,num=n)
print("sequential_array:",sequential_array)

print("--------------------------------------------------")

sequential_array_to_excluded_end = np.linspace(a,b,num=n,endpoint=False)
print("sequential_array_to_excluded_end:",sequential_array_to_excluded_end)

"""

# "Visualisation"

# Visualising the numpy arrays created using "np.linspace".

"""
import matplotlib.pyplot as plt
N = 8
y = np.zeros(N)
x1 = np.linspace(0,10,N,endpoint=True)
x2 = np.linspace(0,10,N,endpoint=False)
plt.plot(x1,y,'o')
plt.plot(x2, y+0.5, 'o')
plt.ylim([-0.5,1])
plt.show()

"""

# np.empty()

# "np.empty" creates an uninitialized empty array of the given shape. It might garbage values.

"""
empty_array = np.empty(4,dtype=int)
print("empty_array:",empty_array)

"""

# np.random.random()

# "np.random.random" creates an array of the given shape with random values in the interval [0,1].

"""
random_valued_array = np.random.random(4)
print("random_valued_array:",random_valued_array)

"""

"""
-->Summary : Creating NumPy Arrays

                # np.array
                # np.zeros, np.ones, np.full
                # np.arange, np.linspace
                # np.empty
                # np.random.random
                
"""