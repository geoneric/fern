#include <Python.h>
#include "fern/python_extension/algorithm/gdal/add_overloads.h"
#include "fern/python_extension/algorithm/gdal/add.h"
#include "fern/python_extension/algorithm/gdal/error.h"
#include "fern/python_extension/algorithm/gdal/util.h"


namespace fern {
namespace python {

PyObject* add_gdal_raster_band_gdal_raster_band(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        gdal_raster_band(object1),
        gdal_raster_band(object2)));
}


PyObject* add_gdal_raster_band_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
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
    return python_object(add(
        python_float(object1),
        gdal_raster_band(object2)));
}


PyObject* add_python_float_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
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


#define ADD_ADD(                                                        \
    type1,                                                              \
    type2)                                                              \
{ PyBinaryAlgorithmKey(WrappedDataType::type1, WrappedDataType::type2), \
    add_##type1##_##type2 },


std::map<PyBinaryAlgorithmKey, PyBinaryAlgorithm> add_overloads{
    ADD_ADD(gdal_raster_band, gdal_raster_band)
    ADD_ADD(gdal_raster_band, python_float)
    ADD_ADD(gdal_raster_band, unsupported)

    ADD_ADD(python_float, gdal_raster_band)
    ADD_ADD(python_float, python_float)
    ADD_ADD(python_float, unsupported)

    ADD_ADD(unsupported, gdal_raster_band)
    ADD_ADD(unsupported, python_float)
    ADD_ADD(unsupported, unsupported)
};

} // namespace python
} // namespace fern
