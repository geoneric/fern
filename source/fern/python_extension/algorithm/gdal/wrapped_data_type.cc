#include "fern/python_extension/algorithm/gdal/wrapped_data_type.h"
#include "fern/python_extension/algorithm/gdal/util.h"


namespace fern {
namespace python {

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

} // namespace python
} // namespace fern
