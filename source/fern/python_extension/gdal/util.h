#pragma once
#include <Python.h>
#include "gdal_priv.h"
#include "fern/core/string.h"


struct PyArrayObject;

namespace fern {
namespace python {

String             to_string           (GDALDataType const& data_type);

bool               is_gdal_raster_band (PyObject* object);

bool               is_python_float     (PyObject* object);

GDALRasterBand*    gdal_raster_band    (PyObject* object);

double             python_float        (PyObject const* object);

PyObject*          python_object       (double value);

PyObject*          python_object       (PyArrayObject* array_object);

} // namespace python
} // namespace fern
