#pragma once
#include "fern/feature/core/raster.h"
#include "customization_point.h"


template<
    typename T,
    size_t nr_dimensions>
size_t size(
   fern::Raster<T, nr_dimensions> const& raster,
   size_t dimension)
{
   return raster.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
value_type<fern::Raster<T, nr_dimensions>> const& get(
    fern::Raster<T, nr_dimensions> const& raster,
    size_t row,
    size_t col)
{
    return raster[row][col];
}

template<
    typename T,
    size_t nr_dimensions>
value_type<fern::Raster<T, nr_dimensions>>& get(
    fern::Raster<T, nr_dimensions>& raster,
    size_t row,
    size_t col)
{
    return raster[row][col];
}
