#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>


class GDALRasterBand;


namespace fern {
namespace python {

PyArrayObject*     slope               (GDALRasterBand* raster_band);

} // namespace python
} // namespace fern
