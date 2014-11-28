#include "fern/python_extension/algorithm/core/util.h"


namespace fern {
namespace python {

bool is_python_bool(
    PyObject* object)
{
    return PyBool_Check(object);
}


bool is_python_int(
    PyObject* object)
{
    return PyInt_Check(object);
}


bool is_python_float(
    PyObject* object)
{
    return PyFloat_Check(object);
}


bool python_bool(
    PyObject const* object)
{
    assert(object);
    assert(object == Py_True || object == Py_False);
    return object == Py_True;
}


int64_t python_int(
    PyObject const* object)
{
    static_assert(sizeof(long) == sizeof(int64_t), "");
    assert(object);
    return PyInt_AS_LONG(const_cast<PyObject*>(object));
}


float64_t python_float(
    PyObject const* object)
{
    assert(object);
    return PyFloat_AS_DOUBLE(const_cast<PyObject*>(object));
}


PyObject* python_object(
    int64_t value)
{
    static_assert(sizeof(long) == sizeof(int64_t), "");
    return PyInt_FromLong(value);
}


PyObject* python_object(
    float64_t value)
{
    return PyFloat_FromDouble(value);
}

} // namespace python
} // namespace fern
