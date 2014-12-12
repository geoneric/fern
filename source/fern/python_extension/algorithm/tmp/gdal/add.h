#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fern/core/types.h"


class GDALRasterBand;


namespace fern {
namespace python {

PyArrayObject*     add                 (int64_t value,
                                        GDALRasterBand* raster_band);

PyArrayObject*     add                 (GDALRasterBand* raster_band,
                                        int64_t value);

PyArrayObject*     add                 (float64_t value,
                                        GDALRasterBand* raster_band);

PyArrayObject*     add                 (GDALRasterBand* raster_band,
                                        float64_t value);

PyArrayObject*     add                 (PyArrayObject* array,
                                        GDALRasterBand* raster_band);

PyArrayObject*     add                 (GDALRasterBand* raster_band,
                                        PyArrayObject* array);

PyArrayObject*     add                 (GDALRasterBand* raster_band1,
                                        GDALRasterBand* raster_band2);

} // namespace python
} // namespace fern
