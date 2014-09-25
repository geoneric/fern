// Example showing how to make it possible to pass instances of user-defined
// classes, unknown to fern, to algorithms.
// - User-defined raster class: example::Raster
// - Glue code to allow passing rasters to algorithm: example::ArgumentTraits
// - Operators to support nice syntax.
#include <cstdlib>
#include <numeric>
#include "fern/example/algorithm/operation.h"
#include "fern/example/algorithm/operator.h"


namespace example {

// Execution policy to use.
fern::ExecutionPolicy execution_policy = fern::parallel;

// In case the execution policy is fern::parallel, we need to create a
// ThreadClient instance. This will, in turn, create a thread pool which is
// used by the parallel algorithms.
// In case the execution policy is fern::sequential, we don't need to create
// a ThreadClient instance. In fact, we shouldn't, for reasons explained in
// the ThreadPool class documentation.
fern::ThreadClient client;

}


int main(
    int /* argc */,
    char** /* argv */)
{
    using namespace example;

    double const cell_size{5.0};
    size_t const nr_rows{6000};
    size_t const nr_cols{4000};

    // [0, 1, 2, 3, ...]
    Raster<int32_t> raster1(cell_size, nr_rows, nr_cols);
    std::iota(raster1.values().begin(), raster1.values().end(), 0);

    // [5, 5, 5, ...]
    Raster<int32_t> raster2(cell_size, nr_rows, nr_cols);
    std::fill(raster2.values().begin(), raster2.values().end(), 5);

    // [5, 6, 7, 8, ...]
    // Operator syntax.
    auto raster3 = raster1 + raster2;
    assert(raster3.values()[100] == 100 + 5);

    // Function call syntax.
    auto raster4 = slope(cast<double>(raster3));

    return EXIT_SUCCESS;
}
