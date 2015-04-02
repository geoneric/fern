#pragma once
#include "type_traits/raster.h"


template<
    typename Raster>
size_t size(Raster const& raster, size_t dimension);

template<
    typename Raster>
value_type<Raster> const& get(Raster const& raster, size_t row, size_t col);

template<
    typename Raster>
value_type<Raster>& get(Raster& raster, size_t row, size_t col);
