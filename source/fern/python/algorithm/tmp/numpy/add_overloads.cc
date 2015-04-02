#include <Python.h>
#include "fern/python_extension/algorithm/numpy/add_overloads.h"
#include "fern/python_extension/algorithm/core/add.h"
#include "fern/python_extension/algorithm/core/util.h"
#include "fern/python_extension/algorithm/numpy/add.h"
#include "fern/python_extension/algorithm/numpy/util.h"


namespace fern {
namespace python {
namespace numpy {

PyObject* add_python_int_python_int(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(core::add(
        python_int(object1),
        python_int(object2)));
}


PyObject* add_python_float_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(core::add(
        python_float(object1),
        python_float(object2)));
}


PyObject* add_python_int_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(core::add(
        python_int(object1),
        python_float(object2)));
}


PyObject* add_python_float_python_int(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(core::add(
        python_int(object2),
        python_float(object1)));
}


PyObject* add_python_int_numpy_array(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        python_int(object1),
        numpy_array(object2)));
}


PyObject* add_numpy_array_python_int(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        numpy_array(object1),
        python_int(object2)));
}


PyObject* add_python_float_numpy_array(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        python_float(object1),
        numpy_array(object2)));
}


PyObject* add_numpy_array_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        numpy_array(object1),
        python_float(object2)));
}


PyObject* add_numpy_array_numpy_array(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        numpy_array(object1),
        numpy_array(object2)));
}


// -----------------------------------------------------------------------------


#define ADD_ADD(                                                       \
    type1,                                                             \
    type2)                                                             \
{ BinaryAlgorithmKey(WrappedDataType::type1, WrappedDataType::type2),  \
    add_##type1##_##type2 },


BinaryOperationMap<BinaryAlgorithmKey> add_overloads{
    ADD_ADD(python_int, python_int)
    ADD_ADD(python_float, python_float)

    ADD_ADD(python_int, python_float)
    ADD_ADD(python_float, python_int)

    ADD_ADD(python_int, numpy_array)
    ADD_ADD(numpy_array, python_int)

    ADD_ADD(python_float, numpy_array)
    ADD_ADD(numpy_array, python_float)

    ADD_ADD(numpy_array, numpy_array)
};


#undef ADD_ADD

} // namespace numpy
} // namespace python
} // namespace fern