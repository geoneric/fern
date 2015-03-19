#pragma once
#include <utility>
#include "fern/core/assert.h"
#include "fern/core/value_type.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/algorithm/core/argument_customization_point.h"
#include "fern/algorithm/core/argument_traits.h"
#include "fern/algorithm/core/cast.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/dont_divide_by_weights.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/divide.h"
#include "fern/algorithm/algebra/elementary/multiply.h"
#include "fern/algorithm/algebra/elementary/subtract.h"
#include "fern/algorithm/algebra/boole/defined.h"
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
        ExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        // TODO Select out-of-range based on input out-of-range policy and/or
        //      output-no-data policy.
        // TODO We handle out-of-range per operation. Analyse the algorithm
        //      and see if we can be more lazy: let infinity and NaN propagate
        //      and only test the result values.
        // TODO Get rid of OutOfRangePolicy? It is not used.

        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

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
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, kernel, result);

        // Calculate the sum of the kernel weights. This equals the
        // convolution of the mask (inverted) by the kernel used above,
        // without normalizing by the kernel weights.
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
        // calculated per cell is sum(kernel) -> 20.
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

        using MultipliedValues = decltype(multiplied_values);

        {
            using INP1 = SkipNoData;
            using INP2 = decltype(std::get<0>(input_no_data_policy));

            InputNoDataPolicies<INP1, INP2> input_no_data_policy_{
                {}, {std::get<0>(input_no_data_policy)}};

            using OutputNoDataPolicyMultipliedValues =
                OutputNoDataPolicyTemporary<OutputNoDataPolicy,
                    MultipliedValues>;

            OutputNoDataPolicyMultipliedValues output_no_data_policy_{
                mask(multiplied_values)};

            algebra::multiply<
                algorithm::multiply::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy_,
                    execution_policy, sum_of_weights, value, multiplied_values);
        }

        // Subtract the convolution result by the multiplied values.
        // Result subtracted_results;
        {
            // Base INP1 on result.
            // Base INP2 on multiplied_values.
            using INP1 = InputNoDataPolicyTemporary<InputNoDataPolicy, Result>;
            using INP2 = InputNoDataPolicyTemporary<InputNoDataPolicy,
                MultipliedValues>;

            InputNoDataPolicies<INP1, INP2> input_no_data_policy_{
                {mask(result)}, {mask(multiplied_values)}};

            algebra::subtract<
                algorithm::subtract::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy,
                    execution_policy, result, multiplied_values, result);
        }

        // Divide subtracted results by the area of the cells.
        {
            // Base INP1 on result.
            using INP1 = InputNoDataPolicyTemporary<InputNoDataPolicy, Result>;
            using INP2 = SkipNoData;

            InputNoDataPolicies<INP1, INP2> input_no_data_policy_{
                {mask(result)}, {}};

            algebra::divide<
                // Cheap, number as denomenator.
                divide::OutOfDomainPolicy,
                divide::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy,
                    execution_policy, result, cell_area(value), result);
        }
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
    ExecutionPolicy& execution_policy,
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
