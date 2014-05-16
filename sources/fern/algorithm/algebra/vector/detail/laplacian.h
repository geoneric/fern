#pragma once
#include "fern/core/value_type.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/dont_divide_by_weights.h"
#include "fern/algorithm/convolution/neighborhood/square.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/divide.h"
#include "fern/algorithm/algebra/elementary/multiply.h"
#include "fern/algorithm/algebra/elementary/subtract.h"
#include "fern/algorithm/statistic/sum.h"


namespace fern {
namespace laplacian {
namespace detail {
namespace dispatch {

template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class RasterCollectionCategory>
class Laplacian
{
};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Laplacian<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        raster_2d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Laplacian()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Laplacian(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 2d raster
    inline void calculate(
        Values const& values,
        Result& result)
    {
        using Weight = value_type<Values>;

        // Convolve using the kernel below, without normalizing the result
        // by the kernel weights..
        fern::Square<Weight, 1> kernel({
            {2, 3, 2},
            {3, 0, 3},
            {2, 3, 2}
        });
        // TODO In case of no-data, the center value must be used.
        convolve<DontDivideByWeights>(values, kernel, result);

        // TODO hier verder
        // // Calculate the sum of the kernel weights.
        // Weight sum_of_weights;
        // sum(kernel, sum_of_weights);

        // // Multiply the values by the sum of weights.
        // Values multiplied_values;
        // multiply(sum_of_weights, values, multiplied_values);

        // // Subtract the convolution result by the multiplied values.
        // Result subtracted_results;
        // subtract(result, multiplied_values, subtracted_results);

        // // Divide subtracted results by the area of the cells.
        // double cell_area = area(values);
        // divide(result, cell_area, result);
    }

};

} // namespace dispatch
} // namespace detail
} // namespace laplacian
} // namespace fern
