// Example showing how to make it possible to pass instances of user-defined
// classes, unknown to fern, to algorithms.
// - User-defined raster class: example::Raster
// - Glue code to allow passing rasters to algorithm: example::ArgumentTraits
// - Add two rasters using default policies.

#include <cstdlib>

// Include the relevant traits before including the algorithm.
#include "fern/example/algorithm/raster_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"


int main(
    int /* argc */,
    char** /* argv */)
{
    using namespace example;

    size_t const nr_rows{600};
    size_t const nr_cols{400};

    Raster<int32_t> raster1(nr_rows, nr_cols);
    Raster<int32_t> raster2(nr_rows, nr_cols);
    Raster<int32_t> raster3(nr_rows, nr_cols);

    // [0, 1, 2, 3, ...]
    std::iota(raster1.values().begin(), raster1.values().end(), 0);

    // [5, 5, 5, ...]
    std::fill(raster2.values().begin(), raster2.values().end(), 5);

    // [5, 6, 7, 8, ...]
    fern::algebra::add(raster1, raster2, raster3);

    assert(raster3.values()[100] == 100 + 5);

    return EXIT_SUCCESS;
}
