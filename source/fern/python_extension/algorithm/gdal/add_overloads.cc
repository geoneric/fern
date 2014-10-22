#include <Python.h>
#include "fern/python_extension/algorithm/gdal/add_overloads.h"
#include "fern/python_extension/algorithm/core/add.h"
#include "fern/python_extension/algorithm/core/util.h"
#include "fern/python_extension/algorithm/numpy/add.h"
#include "fern/python_extension/algorithm/numpy/util.h"
#include "fern/python_extension/algorithm/gdal/add.h"
#include "fern/python_extension/algorithm/gdal/util.h"


namespace fern {
namespace python {

PyObject* add_python_float_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        python_float(object1),
        python_float(object2)));
}


PyObject* add_python_float_numpy_array(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        python_float(object1),
        numpy_array(object2)));
}


PyObject* add_python_float_gdal_raster_band(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        python_float(object1),
        gdal_raster_band(object2)));
}


// -----------------------------------------------------------------------------


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


PyObject* add_numpy_array_gdal_raster_band(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        numpy_array(object1),
        gdal_raster_band(object2)));
}


// -----------------------------------------------------------------------------


PyObject* add_gdal_raster_band_python_float(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        gdal_raster_band(object1),
        python_float(object2)));
}


PyObject* add_gdal_raster_band_numpy_array(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        gdal_raster_band(object1),
        numpy_array(object2)));
}


PyObject* add_gdal_raster_band_gdal_raster_band(
    PyObject* object1,
    PyObject* object2)
{
    return python_object(add(
        gdal_raster_band(object1),
        gdal_raster_band(object2)));
}


// -----------------------------------------------------------------------------


#define ADD_ADD(                                                       \
    type1,                                                             \
    type2)                                                             \
{ BinaryAlgorithmKey(WrappedDataType::type1, WrappedDataType::type2),  \
    add_##type1##_##type2 },

BinaryOperationMap add_overloads{
    ADD_ADD(python_float, python_float)
    ADD_ADD(python_float, numpy_array)
    ADD_ADD(python_float, gdal_raster_band)

    ADD_ADD(numpy_array, python_float)
    ADD_ADD(numpy_array, numpy_array)
    ADD_ADD(numpy_array, gdal_raster_band)

    ADD_ADD(gdal_raster_band, python_float)
    ADD_ADD(gdal_raster_band, numpy_array)
    ADD_ADD(gdal_raster_band, gdal_raster_band)
};

#undef ADD_ADD

} // namespace python
} // namespace fern
