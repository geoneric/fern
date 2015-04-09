// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_traits.h"
#include "fern/algorithm/core/detail/compress.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Copy all non-no-data values from @a value to @a result.

    - Value type of @a Value must be copy-assignable.
    - Value type of @a Value and @a Result must be the same.
    - Result must be a one-dimensional collection.
    - @a result must have the same size as @a value.
    - Count must be integral.

    The @a count returned can be used to resize @a result to the actual
    number of values it contains, e.g. in case result is an std::vector:

    @code
    compress(input_no_data_policy, execution_policy, values, result, count);
    result.resize(count);
    @endcode
*/
template<
    typename InputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result,
    typename Count>
void compress(
    InputNoDataPolicy const& input_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result,
    Count& count)
{
    FERN_STATIC_ASSERT(std::is_copy_assignable, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Value>, value_type<Result>)
    static_assert(rank<Result>() == 1, "");
    FERN_STATIC_ASSERT(std::is_integral, Count)
    assert(size(result) == size(value));

    compress::detail::compress<>(input_no_data_policy, execution_policy,
        value, result, count);
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
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result,
    Count& count)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;

    compress<>(InputNoDataPolicy{{}}, execution_policy, value, result, count);
}

} // namespace core
} // namespace algorithm
} // namespace fern
