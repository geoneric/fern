#include <Python.h>
#include "fern/python_extension/algorithm/numpy/sqrt_overloads.h"
#include "fern/python_extension/algorithm/core/sqrt.h"
#include "fern/python_extension/algorithm/core/util.h"
#include "fern/python_extension/algorithm/numpy/sqrt.h"
#include "fern/python_extension/algorithm/numpy/util.h"


namespace fern {
namespace python {
namespace numpy {

PyObject* sqrt_python_float(
    PyObject* object)
{
    return python_object(core::sqrt(
        python_float(object)));
}


PyObject* sqrt_numpy_array(
    PyObject* object)
{
    return python_object(sqrt(
        numpy_array(object)));
}


// -----------------------------------------------------------------------------


#define ADD_SQRT(                                          \
    type)                                                  \
{ UnaryAlgorithmKey(WrappedDataType::type), sqrt_##type },


UnaryOperationMap sqrt_overloads{
    ADD_SQRT(python_float)
    ADD_SQRT(numpy_array)
};


#undef ADD_SQRT

} // namespace numpy
} // namespace python
} // namespace fern
