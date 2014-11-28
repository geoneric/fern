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


bool is_masked_numpy_array(
    PyObject* object)
{
    bool result = false;

    if(PyObject_HasAttrString(object, "mask") == 1) {
        // New reference.
        PyObject* mask_object_{mask_object(object)};
        assert(mask_object_ != nullptr);

        if(is_numpy_array(mask_object_)) {
           if(PyArray_TYPE(numpy_array(mask_object_)) == NPY_BOOL) {
               result = true;
           }
        }

        Py_DECREF(mask_object_);
    }

    return result;
}


bool is_masked_numpy_array(
    PyArrayObject* object)
{
    return is_masked_numpy_array(python_object(object));
}


PyObject* mask_object(
    PyObject* object)
{
    return PyObject_GetAttrString(object, "mask");
}


PyObject* mask_object(
    PyArrayObject* array_object)
{
    return mask_object(python_object(array_object));
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
