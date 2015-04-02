#include <cstddef>
#include "customization_point/raster.h"
#include "policy/detect_no_data_by_value.h"
#include "policy/input_no_data_policies.h"
#include "policy/mark_no_data_by_value.h"
#include "policy/skip_no_data.h"
#include "pow.h"


int main(
    int /* argc */,
    char** /* argv */)
{
    using Raster = fern::Raster<double, 2>;
    using RasterInputNoDataPolicy = DetectNoDataByValue<Raster>;
    using NumberInputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = MarkNoDataByValue<Raster>;

    double no_data_marker{-999};

    size_t const nr_rows{2};
    size_t const nr_cols{3};
    double west{0.0};
    double north{0.0};
    double cell_size{10.0};
    Raster::Transformation transformation{{west, cell_size, north, cell_size}};
    Raster base(fern::extents[nr_rows][nr_cols], transformation);
    // Assign values to base raster.
    // ...

    // pow(raster, number, raster)
    {
        double exponent{2};
        Raster result(fern::extents[nr_rows][nr_cols], transformation);

        InputNoDataPolicies<RasterInputNoDataPolicy, NumberInputNoDataPolicy>
            input_no_data_policy({base, no_data_marker}, {});
        OutputNoDataPolicy output_no_data_policy{result, no_data_marker};

        pow(input_no_data_policy, output_no_data_policy, base, exponent,
            result);
    }

    // pow(raster, raster, raster)
    {
        Raster exponent(fern::extents[nr_rows][nr_cols], transformation);
        // Assign values to exponent raster.
        // ...
        Raster result(fern::extents[nr_rows][nr_cols], transformation);

        InputNoDataPolicies<RasterInputNoDataPolicy, RasterInputNoDataPolicy>
            input_no_data_policy{{base, no_data_marker},
                {exponent, no_data_marker}};
        OutputNoDataPolicy output_no_data_policy{result, no_data_marker};

        pow(input_no_data_policy, output_no_data_policy, base, exponent,
            result);
    }

    return EXIT_SUCCESS;
}
