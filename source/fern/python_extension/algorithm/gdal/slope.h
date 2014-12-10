#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>


class GDALRasterBand;


namespace fern {
namespace python {
namespace gdal {

PyArrayObject*     slope               (GDALRasterBand* raster_band);

} // namespace gdal
} // namespace python
} // namespace fern
