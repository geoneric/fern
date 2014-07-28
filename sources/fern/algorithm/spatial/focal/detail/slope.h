#pragma once
#include "fern/core/argument_categories.h"
#include "fern/core/assert.h"
#include "fern/core/value_type.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/policies.h"


namespace fern {
namespace slope {
namespace detail {
namespace dispatch {

template<
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
class Slope
{
};


template<
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
struct Slope<
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

        using Float = value_type<Value>;
        FERN_STATIC_ASSERT(std::is_floating_point, Float)

        // dz_dx: convolve using this kernel.
        Square<Float, 1> dz_dx_kernel({
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1}
        });

        // dz_dx: convolve using this kernel.
        Square<Float, 1> dz_dy_kernel({
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        });

        auto extents = fern::extents[size(value, 0)][size(value, 1)];

        MaskedArray<Float, 2> dz_dx(extents);
        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, dz_dx_kernel, dz_dx);

        algebra::divide<
            fern::divide::OutOfDomainPolicy,  // TODO Pick correct policy.
            fern::divide::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dx, 8 * cell_size(value, 0), dz_dx);

        MaskedArray<Float, 2> dz_dy(extents);
        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, dz_dy_kernel, dz_dy);

        algebra::divide<
            fern::divide::OutOfDomainPolicy,  // TODO Pick correct policy.
            fern::divide::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dy, 8 * cell_size(value, 1), dz_dy);

        algebra::pow<
            pow::OutOfDomainPolicy,  // TODO Pick correct policy.
            pow::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dx, Float(2), dz_dx);

        algebra::pow<
            pow::OutOfDomainPolicy,  // TODO Pick correct policy.
            pow::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dy, Float(2), dz_dy);

        algebra::add<
            fern::add::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dx, dz_dy, result);

        // TODO Whether or not to detect out of domain values depends on
        //      output no-data policy passed in. Will sqrt succeed if input
        //      < 0?
        algebra::sqrt<sqrt::OutOfDomainPolicy>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            result, result);
    }

};

} // namespace dispatch


template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void slope(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    dispatch::Slope<
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
} // namespace slope
} // namespace fern
