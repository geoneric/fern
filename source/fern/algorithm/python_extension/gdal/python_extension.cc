#include <Python.h>
#include <functional>
#include <iostream>
#include <map>
#include "gdal_priv.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/python_extension/swig_runtime.h"
#include "fern/algorithm/python_extension/gdal/algorithm.h"
#include "fern/algorithm/python_extension/gdal/error.h"


namespace fern {

SwigPyObject* swig_object(
    PyObject* object)
{
    return SWIG_Python_GetSwigThis(object);
}


SwigPyObject* swig_object(
    PyObject* object,
    std::string const& typename_)
{
    SwigPyObject* result = swig_object(object);

    if(result != nullptr && std::string(result->ty->name) == typename_) {
        return result;
    }

    return nullptr;
}


bool is_gdal_raster_band(
    PyObject* object)
{
    return swig_object(object, "_p_GDALRasterBandShadow") != nullptr;
}


bool is_python_float(
    PyObject* object)
{
    return PyFloat_Check(object);
}


GDALRasterBand* gdal_raster_band(
    PyObject* object)
{
    assert(object);
    return static_cast<GDALRasterBand*>(swig_object(object)->ptr);
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


PyObject* python_object(
    PyArrayObject* array_object)
{
    assert(array_object);
    return reinterpret_cast<PyObject*>(array_object);
}


// Types of data supported by this module.
enum class WrappedDataType
{
    // TODO python_int,
    // TODO python_long,
    python_float,
    gdal_raster_band,
    // TODO numpy_array,  -> Forward to numpy module as much as possible.
    // TODO numpy_int8,  -> ..
    // TODO numpy_int16,
    // TODO numpy_int32,
    // TODO numpy_int64,
    // TODO numpy_uint8,
    // TODO numpy_uint16,
    // TODO numpy_uint32,
    // TODO numpy_uint64,
    // TODO numpy_float32,
    // TODO numpy_float64,
    unsupported
};


WrappedDataType data_type(
    PyObject* object)
{
    WrappedDataType result{WrappedDataType::unsupported};

    if(is_python_float(object)) {
        result = WrappedDataType::python_float;
    }
    else if(is_gdal_raster_band(object)) {
        result = WrappedDataType::gdal_raster_band;
    }

    return result;
}


using PyBinaryAlgorithm = std::function<PyObject*(PyObject*, PyObject*)>;
using PyBinaryAlgorithmKey = std::tuple<WrappedDataType, WrappedDataType>;


static std::map<PyBinaryAlgorithmKey, PyBinaryAlgorithm> add_overloads;


PyObject* add_gdal_raster_band_gdal_raster_band(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(fern::add(
        gdal_raster_band(object1),
        gdal_raster_band(object2)));
}


PyObject* add_gdal_raster_band_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(fern::add(
        gdal_raster_band(object1),
        python_float(object2)));
}


void raise_unsupported_argument_type_exception(
    PyObject* object)
{
    PyObject* type_object = reinterpret_cast<PyObject*>(object->ob_type);
    PyObject* representation_object = PyObject_Repr(type_object);

    raise_unsupported_argument_type_exception(PyString_AsString(
        representation_object));
}


PyObject* add_gdal_raster_band_unsupported(
    PyObject* /* object1 */,
    PyObject* object2)
{
    raise_unsupported_argument_type_exception(object2);
    return nullptr;
}


PyObject* add_python_float_gdal_raster_band(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(fern::add(
        python_float(object1),
        gdal_raster_band(object2)));
}


PyObject* add_python_float_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(fern::add(
        python_float(object1),
        python_float(object2)));
}


PyObject* add_python_float_unsupported(
    PyObject* /* object1 */,
    PyObject* object2)
{
    raise_unsupported_argument_type_exception(object2);
    return nullptr;
}


PyObject* add_unsupported_gdal_raster_band(
    PyObject* object1,
    PyObject* /* object2 */)
{
    raise_unsupported_argument_type_exception(object1);
    return nullptr;
}


PyObject* add_unsupported_python_float(
    PyObject* object1,
    PyObject* /* object2 */)
{
    raise_unsupported_argument_type_exception(object1);
    return nullptr;
}


PyObject* add_unsupported_unsupported(
    PyObject* object1,
    PyObject* /* object2 */)
{
    raise_unsupported_argument_type_exception(object1);
    return nullptr;
}


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
    {"add", add, METH_VARARGS,
        // TODO
        "Bladiblah from gdal"},
    {nullptr, nullptr, 0, nullptr}
};

} // namespace fern


#define ADD_ADD(                                                            \
        type1,                                                              \
        type2)                                                              \
    fern::add_overloads.emplace(std::make_pair(fern::PyBinaryAlgorithmKey(  \
            fern::WrappedDataType::type1, fern::WrappedDataType::type2),    \
        fern::add_##type1##_##type2));

#define PERMUTATE(                             \
        macro)                                 \
    macro(gdal_raster_band, gdal_raster_band)  \
    macro(gdal_raster_band, python_float)      \
    macro(gdal_raster_band, unsupported)       \
                                               \
    macro(python_float, gdal_raster_band)      \
    macro(python_float, python_float)          \
    macro(python_float, unsupported)           \
                                               \
    macro(unsupported, gdal_raster_band)       \
    macro(unsupported, python_float)           \
    macro(unsupported, unsupported)


static fern::ThreadClient thread_client;


PyMODINIT_FUNC initfern_algorithm_gdal(
    void)
{
    PyObject* module = Py_InitModule("fern_algorithm_gdal", fern::methods);

    if(module) {
        PERMUTATE(ADD_ADD)
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
