#pragma once
#include "fern/core/argument_traits.h"
#include "fern/algorithm/core/detail/compress.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Copy all non-no-data values from \a value to \a result.

    - Value type of @Value must be copy-assignable.
    - Value type of @Value and @Result must be the same.
    - Result must be a one-dimensional collection.
    - Count must be integral.
*/
template<
    typename InputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result,
    typename Count>
void compress(
    InputNoDataPolicy const& input_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result,
    Count& count)
{
    FERN_STATIC_ASSERT(std::is_copy_assignable, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Value>, value_type<Result>)
    static_assert(rank<Result>() == 1, "");
    FERN_STATIC_ASSERT(std::is_integral, Count)

    compress::detail::compress<>(input_no_data_policy, execution_policy,
        value, result, count);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename InputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result,
    typename Count>
void compress(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result,
    Count& count)
{
    compress<>(InputNoDataPolicy(), execution_policy, value, result, count);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result,
    typename Count>
void compress(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result,
    Count& count)
{
    using InputNoDataPolicy = SkipNoData<>;

    compress<>(InputNoDataPolicy(), execution_policy, value, result, count);
}

} // namespace core
} // namespace algorithm
} // namespace fern
