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
#include "fern/algorithm/core/detail/decompress.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Copy all non-no-data values from @a value to @a result, while
                inserting no-data given the @a input_no_data_policy.

    - Value type of @a Value must be copy-assignable.
    - Value type of @a Value and @a Result must be the same.
    - @a Value must be a one-dimensional collection.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void decompress(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_copy_assignable, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Value>, value_type<Result>)
    static_assert(rank<Value>() == 1, "");
    assert(size(value) <= size(result));

    decompress::detail::decompress<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void decompress(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    decompress<>(InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern
