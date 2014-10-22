#pragma once
#include <Python.h>
#include <gdal_priv.h>
#include "fern/core/string.h"


namespace fern {
namespace python {

String             to_string           (GDALDataType const& data_type);

bool               is_gdal_raster_band (PyObject* object);

GDALRasterBand*    gdal_raster_band    (PyObject* object);

} // namespace python
} // namespace fern
