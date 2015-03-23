#include <cmath>
#include "fern/feature/core/raster.h"


fern::Raster<double, 2> pow(
    fern::Raster<double, 2> const& base,
    double exponent)
{
    size_t const nr_rows{base.shape()[0]};
    size_t const nr_cols{base.shape()[1]};
    fern::Raster<double, 2> result(fern::extents[nr_rows][nr_cols],
        base.transformation());

    for(size_t r = 0; r < nr_rows; ++r) {
        for(size_t c = 0; c < nr_cols; ++c) {
            result[r][c] = std::pow(base[r][c], exponent);
        }
    }

    return result;
}
