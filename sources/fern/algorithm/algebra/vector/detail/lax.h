#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/clone.h"
#include "fern/core/value_type.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/algorithm/algebra/elementary/multiply.h"


namespace fern {
namespace lax {
namespace detail {
namespace dispatch {

template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
class Lax
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
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
        ExecutionPolicy const& execution_policy,
        Value const& value,
        value_type<Value> const& fraction,
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

        // result = (1 - f) * value + f * convolution(value, kernel);

        convolution::convolve<
            convolve::SkipNoData,
            convolve::DivideByWeights,
            convolve::SkipOutOfImage,
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, kernel, result);

        algebra::multiply<
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            fern::multiply::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                fraction, result, result);


        auto multiplied_values = clone<Float>(result);
        algebra::multiply<
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            fern::multiply::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                (1.0 - fraction), value, multiplied_values);

        algebra::add<
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            fern::add::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                multiplied_values, result, result);
    }

};

} // namespace dispatch


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void lax(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
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
} // namespace fern
