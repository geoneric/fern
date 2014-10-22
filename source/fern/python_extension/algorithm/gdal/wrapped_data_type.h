#pragma once
#include <Python.h>


namespace fern {
namespace python {

//! Types of data supported by this module.
enum class WrappedDataType
{
    // TODO python_int,
    // TODO python_long,
    python_float,
    numpy_array,
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


WrappedDataType    data_type           (PyObject* object);

} // namespace python
} // namespace fern
