import cython
import numpy as np
cimport numpy as np


# Declare the interface to the C++ code.
cdef extern from "fern_algorithm_numpy.h" namespace "fern": \
    PyArrayObject* addd(PyArrayObject const* array, PyFloatObject const* value)



#   PyArrayObject* add(PyArrayObject*, PyFloatObject*)




