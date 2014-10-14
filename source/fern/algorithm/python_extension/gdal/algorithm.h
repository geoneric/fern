#pragma once
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fern/algorithm/python_extension/swig_runtime.h"


class GDALRasterBand;


namespace fern {

PyArrayObject*     add                 (GDALRasterBand const* raster_band1,
                                        GDALRasterBand const* raster_band2);

PyArrayObject*     add                 (GDALRasterBand const* raster_band,
                                        double float_);

PyArrayObject*     add                 (double float_,
                                        GDALRasterBand const* raster_band);

double             add                 (double float1,
                                        double float2);

} // namespace fern
