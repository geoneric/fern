#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fern/core/types.h"


class GDALRasterBand;


namespace fern {
namespace python {

PyArrayObject*     add                 (GDALRasterBand* raster_band1,
                                        GDALRasterBand* raster_band2);

PyArrayObject*     add                 (GDALRasterBand* raster_band,
                                        float64_t value);

PyArrayObject*     add                 (float64_t value,
                                        GDALRasterBand* raster_band);

double             add                 (float64_t value1,
                                        float64_t value2);

} // namespace python
} // namespace fern
