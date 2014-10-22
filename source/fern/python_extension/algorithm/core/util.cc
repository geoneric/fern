#include "fern/python_extension/algorithm/core/util.h"


namespace fern {
namespace python {

bool is_python_float(
    PyObject* object)
{
    return PyFloat_Check(object);
}


double python_float(
    PyObject const* object)
{
    assert(object);
    return PyFloat_AS_DOUBLE(const_cast<PyObject*>(object));
}


PyObject* python_object(
    double value)
{
    return PyFloat_FromDouble(value);
}

} // namespace python
} // namespace fern
