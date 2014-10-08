#pragma once
#include <utility>
#include "fern/core/assert.h"
#include "fern/core/clone.h"
#include "fern/core/value_type.h"
#include "fern/feature/core/array_traits.h"
#include "fern/algorithm/core/cast.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/dont_divide_by_weights.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/divide.h"
#include "fern/algorithm/algebra/elementary/multiply.h"
#include "fern/algorithm/algebra/elementary/subtract.h"
#include "fern/algorithm/algebra/boolean/defined.h"
#include "fern/algorithm/statistic/sum.h"
#include "fern/algorithm/statistic/unary_max.h"


namespace fern {
namespace algorithm {
namespace laplacian {
namespace detail {
namespace dispatch {

template<
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
class Laplacian
{
};


template<
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
struct Laplacian<
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        // TODO See also comments in detail/slope.h.

        using Float = value_type<Value>;
        FERN_STATIC_ASSERT(std::is_floating_point, Float)

        // Convolve using the kernel below, without normalizing the result
        // by the kernel weights.
        Square<Float, 1> kernel({
            {2, 3, 2},
            {3, 0, 3},
            {2, 3, 2}
        });

        convolution::convolve<
            convolve::SkipNoData,
            convolve::DontDivideByWeights,
            convolve::SkipOutOfImage,
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, kernel, result);

        // Calculate the sum of the kernel weights. This equals the convolution
        // of the mask (inverted) by the kernel used above, without
        // normalizing by the kernel weights.
        auto extents = fern::extents[size(value, 0)][size(value, 1)];

        // Determine which cells have a valid (non-no-data) value. The result
        // will not contain no-data. All elements have a valid value (true or
        // false.
        Array<bool, 2> defined(extents);
        algebra::defined(input_no_data_policy, execution_policy, defined);

        // Cast array of bool to array of floats. It is not needed to take
        // no-data and range errors into account.
        Array<Float, 2> defined_as_floats(extents);
        core::cast<>(execution_policy, defined, defined_as_floats);

        // It is not needed to take range errors into account. The max value
        // calculate per cells is sum(kernel) -> 20.
        Array<Float, 2> sum_of_weights(extents);
        convolution::convolve<
            convolve::SkipNoData,
            convolve::DontDivideByWeights,
            convolve::SkipOutOfImage,
            unary::DiscardRangeErrors>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                defined_as_floats, kernel, sum_of_weights);

#ifndef NDEBUG
        {
            Float max;
            statistic::unary_max(execution_policy, sum_of_weights, max);
            assert(max <= 20.0);
        }
#endif

        // Multiply the values by the sum of weights.
        auto multiplied_values = clone<Float>(result);
        algebra::multiply<
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            algorithm::multiply::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                sum_of_weights, value, multiplied_values);

        // Subtract the convolution result by the multiplied values.
        // Result subtracted_results;
        algebra::subtract<
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            algorithm::subtract::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                result, multiplied_values, result);

        // Divide subtracted results by the area of the cells.
        algebra::divide<
            // TODO: Select OutOfDomain policy based on the
            //       output-no-data-policy passed in.
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            algorithm::divide::OutOfDomainPolicy,
            algorithm::divide::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                result, cell_area(value), result);
    }

};

} // namespace dispatch


template<
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void laplacian(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    dispatch::Laplacian<
        OutOfRangePolicy<value_type<Value>, value_type<Result>>,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace laplacian
} // namespace algorithm
} // namespace fern
