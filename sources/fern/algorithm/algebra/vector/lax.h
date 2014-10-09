#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/vector/detail/lax.h"


namespace fern {
namespace algorithm {
namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @brief      Calculate the lax of @a value and write the result to
                @a result.

    The algorithm implemented is similar to [the one implemented in PCRaster]
    (https://sourceforge.net/p/pcraster/pcrtree2/ci/master/tree/sources/calc/vf.c).

    In short/pseudo code, the algorithm:

    @code
    result = (1 - fraction) * value + fraction * convolution(value, kernel);
    @endcode

    Kernel:

    @code
    +---+---+---+
    | 2 | 3 | 2 |
    +---+---+---+
    | 3 | 0 | 3 |
    +---+---+---+
    | 2 | 3 | 2 |
    +---+---+---+
    @endcode

    The value type of @a Value and @a Result must be floating point and the
    same.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void lax(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    value_type<Value> const& fraction,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    lax::detail::lax(input_no_data_policy,
        output_no_data_policy, execution_policy, value, fraction, result);
}


/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @overload
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void lax(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    value_type<Value> const& fraction,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    lax(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, fraction, result);
}


/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void lax(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    value_type<Value> const& fraction,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    lax(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, fraction, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
