#include <cstddef>
#include "customization_point/raster.h"
#include "pow.h"


int main(
    int /* argc */,
    char** /* argv */)
{
    using Raster = fern::Raster<double, 2>;

    size_t const nr_rows{2};
    size_t const nr_cols{3};
    double west{0.0};
    double north{0.0};
    double cell_size{10.0};
    Raster::Transformation transformation{{west, cell_size, north, cell_size}};
    Raster base(fern::extents[nr_rows][nr_cols], transformation);
    // Assign values to base raster.
    // ...

    Raster result(fern::extents[nr_rows][nr_cols], transformation);

    // pow(raster, number, raster)
    {
        double exponent{2};
        pow(base, exponent, result);
    }

    // pow(raster, raster, raster)
    {
        Raster exponent(fern::extents[nr_rows][nr_cols], transformation);
        // Assign values to exponent raster.
        // ...
        pow(base, exponent, result);
    }

    return EXIT_SUCCESS;
}
