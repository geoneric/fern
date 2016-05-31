// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <type_traits>
#include "fern/core/argument_categories.h"
#include "fern/core/assert.h"
#include "fern/core/value_type.h"
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/policies.h"
#include "fern/algorithm/core/argument_traits.h"


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

        // dz_dx: convolve using this kernel.
        Square<Float, 1> dz_dx_kernel({
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1}
        });

        // dz_dy: convolve using this kernel.
        Square<Float, 1> dz_dy_kernel({
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        });

        auto dz_dx(clone<Float>(value));

        using DzDx = decltype(dz_dx);
        using OutputNoDataPolicyDzDx = OutputNoDataPolicyTemporary<
            OutputNoDataPolicy, DzDx>;

        OutputNoDataPolicyDzDx output_no_data_policy_dz_dx{mask(dz_dx)};

        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::KeepNoDataFocusElement,
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy_dz_dx,
                execution_policy, value, dz_dx_kernel, dz_dx);

        using InputNoDataPolicyDzDx = InputNoDataPolicyTemporary<
            InputNoDataPolicy, DzDx>;

        InputNoDataPolicyDzDx input_no_data_policy_dz_dx{mask(dz_dx)};

        {
            InputNoDataPolicies<InputNoDataPolicyDzDx, SkipNoData>
                input_no_data_policy_{input_no_data_policy_dz_dx, {}};

            algebra::divide<
                divide::OutOfDomainPolicy,  // Cheap, number as denomenator.
                divide::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy_dz_dx,
                    execution_policy, dz_dx, static_cast<Float>(8 *
                        cell_size(value, 0)), dz_dx);
        }

        auto dz_dy(clone<Float>(value));

        using DzDy = decltype(dz_dy);
        using OutputNoDataPolicyDzDy = OutputNoDataPolicyTemporary<
            OutputNoDataPolicy, DzDy>;

        OutputNoDataPolicyDzDy output_no_data_policy_dz_dy{mask(dz_dy)};

        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::KeepNoDataFocusElement,
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy_dz_dy,
                execution_policy, value, dz_dy_kernel, dz_dy);

        using InputNoDataPolicyDzDy = InputNoDataPolicyTemporary<
            InputNoDataPolicy, DzDy>;

        InputNoDataPolicyDzDy input_no_data_policy_dz_dy{mask(dz_dy)};

        {
            InputNoDataPolicies<InputNoDataPolicyDzDy, SkipNoData>
                input_no_data_policy_{input_no_data_policy_dz_dy, {}};

            algebra::divide<
                divide::OutOfDomainPolicy,  // Cheap, number as denomenator.
                divide::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy_dz_dy,
                    execution_policy, dz_dy, static_cast<Float>(8 *
                        cell_size(value, 1)), dz_dy);
        }

        {
            InputNoDataPolicies<InputNoDataPolicyDzDx, SkipNoData>
                input_no_data_policy_{input_no_data_policy_dz_dx, {}};

            algebra::pow<
                binary::DiscardDomainErrors,  // Never a domain error.
                pow::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy_dz_dx,
                    execution_policy, dz_dx, Float(2), dz_dx);
        }

        {
            InputNoDataPolicies<InputNoDataPolicyDzDy, SkipNoData>
                input_no_data_policy_{input_no_data_policy_dz_dy, {}};

            algebra::pow<
                binary::DiscardDomainErrors,  // Never a domain error.
                pow::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy_dz_dy,
                    execution_policy, dz_dy, Float(2), dz_dy);
        }

        {
            InputNoDataPolicies<InputNoDataPolicyDzDx, InputNoDataPolicyDzDy>
                input_no_data_policy_{input_no_data_policy_dz_dx,
                    input_no_data_policy_dz_dy};

            algebra::add<
                add::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy,
                    execution_policy, dz_dx, dz_dy, result);
        }

        {
            using InputNoDataPolicyResult = InputNoDataPolicyTemporary<
                InputNoDataPolicy, Result>;

            InputNoDataPolicyResult input_no_data_policy_result{mask(result)};
            InputNoDataPolicies<InputNoDataPolicyResult>
                input_no_data_policy_{input_no_data_policy_result};

            algebra::sqrt<
                unary::DiscardDomainErrors>(  // Argument >= 0.
                input_no_data_policy_, output_no_data_policy, execution_policy,
                result, result);
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
void slope(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
