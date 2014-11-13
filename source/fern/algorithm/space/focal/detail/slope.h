#pragma once
#include "fern/core/argument_categories.h"
#include "fern/core/assert.h"
#include "fern/core/value_type.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/policies.h"


namespace fern {
namespace algorithm {
namespace slope {
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
class Slope
{
};


template<
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
struct Slope<
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    raster_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        // TODO The implementation of slope uses other algorithms. These other
        //      algorithms have out-of-domain and out-of-range policies, and
        //      currently, the algorithm-specific policies are used. But the
        //      default, non-checking policies could also be used, but then
        //      we don't know of out-of-domain/out-of-range situation occured
        //      during the calculations.
        //      How to choose which policies to use for the algorithms we call
        //      here? If slope's output_no_data_policy doesn't do anything, it
        //      makes no sense to check for out-of-domain/out-of-range, because
        //      we can't mark this in the result.
        //      Slope itself doesn't have out-of-domain policy.
        //      Slope has an out-of-range policy, but it isn't used yet.

        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

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

        auto dz_dx(clone<Float>(value));
        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, dz_dx_kernel, dz_dx);

        algebra::divide<
            divide::OutOfDomainPolicy,  // TODO Pick correct policy.
            divide::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dx, static_cast<Float>(8 * cell_size(value, 0)), dz_dx);

        auto dz_dy(clone<Float>(value));
        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, dz_dy_kernel, dz_dy);

        algebra::divide<
            divide::OutOfDomainPolicy,  // TODO Pick correct policy.
            divide::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dy, static_cast<Float>(8 * cell_size(value, 1)), dz_dy);

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
            add::OutOfRangePolicy>(  // TODO Pick correct policy.
                input_no_data_policy, output_no_data_policy, execution_policy,
                dz_dx, dz_dy, result);

        // TODO Whether or not to detect out of domain values depends on
        //      output no-data policy passed in. Will sqrt succeed if input
        //      < 0?
        algebra::sqrt<sqrt::OutOfDomainPolicy>(  // TODO Pick correct policy.
            input_no_data_policy, output_no_data_policy, execution_policy,
            result, result);
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
        base_class<argument_category<Value>, raster_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace slope
} // namespace algorithm
} // namespace fern
