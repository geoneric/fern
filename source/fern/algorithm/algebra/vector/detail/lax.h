#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/value_type.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/algorithm/algebra/elementary/multiply.h"


namespace fern {
namespace algorithm {
namespace lax {
namespace detail {
namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
class Lax
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
struct Lax<
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
        value_type<Value> const& fraction,
        Result& result)
    {
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

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

        // result = (1 - f) * value + f * convolution(value, kernel);

        // result = convolve(value, kernel)
        convolution::convolve<
            convolve::SkipNoData,
            convolve::DivideByWeights,
            convolve::SkipOutOfImage,
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, kernel, result);

        // result = f * result
        {
            // TODO Tricky, we should use an input no-data policy that can
            //      detect the no-data in the temp result of the convolve.
            //      Like it is now, we won't detect generated no-data, but
            //      only no-data in the origin value passed in. Update INP2.
            //      Given a type, we need to be able to create an
            //      input-no-data policy.
            using INP1 = SkipNoData;
            using INP2 = decltype(std::get<0>(input_no_data_policy));

            InputNoDataPolicies<INP1, INP2> input_no_data_policy_{{},
                {std::get<0>(input_no_data_policy)}};

            algebra::multiply<
                // TODO: Select OutOfRange policy based on the
                //       output-no-data-policy passed in.
                algorithm::multiply::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy,
                    execution_policy, fraction, result, result);
        }

        // tmp = (1.0 - f) * value
        auto multiplied_values = clone<Float>(result);
        {
            using INP1 = SkipNoData;
            using INP2 = decltype(std::get<0>(input_no_data_policy));

            InputNoDataPolicies<INP1, INP2> input_no_data_policy_{{},
                {std::get<0>(input_no_data_policy)}};

            algebra::multiply<
                // TODO: Select OutOfRange policy based on the
                //       output-no-data-policy passed in.
                algorithm::multiply::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy,
                    execution_policy, (1.0 - fraction), value,
                    multiplied_values);
        }

        // result = tmp + result
        {
            // TODO See comment above.
            using INP1 = SkipNoData;
            using INP2 = decltype(std::get<0>(input_no_data_policy));

            InputNoDataPolicies<INP1, INP2> input_no_data_policy_{{},
                {std::get<0>(input_no_data_policy)}};

            algebra::add<
                // TODO: Select OutOfRange policy based on the
                //       output-no-data-policy passed in.
                algorithm::add::OutOfRangePolicy>(
                    input_no_data_policy_, output_no_data_policy,
                    execution_policy, multiplied_values, result, result);
        }
    }

};

} // namespace dispatch


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void lax(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    value_type<Value> const& fraction,
    Result& result)
{
    dispatch::Lax<
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, fraction, result);
}

} // namespace detail
} // namespace lax
} // namespace algorithm
} // namespace fern
