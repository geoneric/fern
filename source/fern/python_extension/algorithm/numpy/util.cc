#include "fern/python_extension/algorithm/numpy/util.h"
#include <map>


namespace fern {
namespace python {

static void init_numpy()
{
    import_array();
}


bool is_numpy_array(
    PyObject* object)
{
    init_numpy();
    return PyArray_Check(object);
}


PyArrayObject* numpy_array(
    PyObject* object)
{
    assert(object);
    return reinterpret_cast<PyArrayObject*>(object);
}


PyObject* python_object(
    PyArrayObject* array_object)
{
    assert(array_object);
    return reinterpret_cast<PyObject*>(array_object);
}

} // namespace python
} // namespace fern
