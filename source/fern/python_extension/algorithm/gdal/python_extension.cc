#include <Python.h>
#include "fern/core/thread_client.h"
#include "fern/python_extension/algorithm/gdal/add_overloads.h"


namespace fern {
namespace python {

static PyObject* add(
    PyObject* /* self */,
    PyObject* arguments)
{
    PyObject* value1_object;
    PyObject* value2_object;

    if(!PyArg_ParseTuple(arguments, "OO", &value1_object, &value2_object)) {
        return nullptr;
    }

    PyObject* result{nullptr};

    try {
        PyBinaryAlgorithmKey key(data_type(value1_object),
            data_type(value2_object));
        result = add_overloads[key](value1_object, value2_object);
        assert((PyErr_Occurred() && result == nullptr) ||
            (!PyErr_Occurred() && result != nullptr));
    }
    catch(std::runtime_error const& exception) {
        PyErr_SetString(PyExc_RuntimeError, exception.what());
        assert(result == nullptr);
    }
    catch(std::exception const& exception) {
        PyErr_SetString(PyExc_StandardError, exception.what());
        assert(result == nullptr);
    }

    assert((PyErr_Occurred() && result == nullptr) ||
        (!PyErr_Occurred() && result != nullptr));
    assert(result != Py_None);
    return result;
}


static PyMethodDef methods[] = {
    {"add", add, METH_VARARGS, "Add two arguments and return the result"},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace python
} // namespace fern


static fern::ThreadClient thread_client;


PyMODINIT_FUNC init_fern_algorithm_gdal(
    void)
{
    PyObject* module = Py_InitModule("_fern_algorithm_gdal",
        fern::python::methods);

    if(module) {
        // ...
    }
}


/// bool is_gdal_dataset(
///     PyObject* object)
/// {
///     return swig_object(object, "_p_GDALDatasetShadow") != nullptr;
/// }


// GDALDataset* extract_gdal_dataset(
//     PyObject* object)
// {
//     // Iff object is a PyObject created by a SWIG wrapped Python extension
//     // (like the osgeo extension), then it has an attribute called 'this'.
//     // This attribute is a SWIG wrapper of type SwigPyObject (a shadow
//     // object, see swig_runtime.h). The shadow object contains a pointer to
//     // a wrapped instance.
// 
//     SwigPyObject* swig_object = SWIG_Python_GetSwigThis(object);
// 
//     if(swig_object == nullptr) {
//         return nullptr;
//     }
// 
//     assert(std::string(swig_object->ty->name) == "_p_GDALDatasetShadow");
//     assert(std::string(SWIG_TypePrettyName(swig_object->ty)) ==
//         "GDALDatasetShadow *");
// 
//     GDALDataset* gdal_dataset = static_cast<GDALDataset*>(swig_object->ptr);
// 
//     return gdal_dataset;
// }


/// swig_type_info* swig_type_info(
///     PyObject* object)
/// {
///     SwigPyObject* result = swig_object(object);
///     assert(result != nullptr);
///     return result->ty;
/// }
