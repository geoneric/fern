// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/detail/offset.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Offset the elements in @a value by @a offset_ and write
                the result to @a result.

    Elements in @a result that have no corresponding value in @a value, will
    be handled by @a output_no_data_policy.

    A positive offset will offset the elements towards higher indices. In case
    of a 1D array, for example, a positive offset will offset the elements
    towards the end of the array.

    - The dimensionality of @a value must be larger than 0.
    - The @a offset_ must have the same dimensionality as @a value.
    - The value type of @a Result must be equal to @a Value.
    - The value type of @a Offset must be signed integral.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Offset,
    typename Result>
void offset(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Offset const& offset_,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_integral, value_type<Offset>)
    FERN_STATIC_ASSERT(std::is_signed, value_type<Offset>)
    static_assert(rank<Value>() > 0, "Rank must be larger than 0");
    static_assert(rank<Value>() == rank<Offset>(),
        "Rank of offset must equal rank of value");

    offset::detail::offset<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, offset_, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Offset,
    typename Result>
void offset(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Offset const& offset_,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    offset<>(InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, offset_, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @brief      Offset the elements in @a value by @a offset_ and write
                the result to @a result.

    Elements in @a result that have no corresponding value in @a value, will
    be assigned the @a fill_value.

    A positive offset will offset the elements towards higher indices. In case
    of a 1D array, for example, a positive offset will offset the elements
    towards the end of the array.

    - The dimensionality of @a value must be larger than 0.
    - The @a offset_ must have the same dimensionality as @a value.
    - The value type of @a Result must be equal to @a Value.
    - The value type of @a Offset must be signed integral.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Offset,
    typename Result>
void offset(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Offset const& offset_,
    value_type<Result> const& fill_value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_integral, value_type<Offset>)
    FERN_STATIC_ASSERT(std::is_signed, value_type<Offset>)
    static_assert(rank<Value>() > 0, "Rank must be larger than 0");
    static_assert(rank<Value>() == rank<Offset>(),
        "Rank of offset must equal rank of value");

    offset::detail::offset<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, offset_, fill_value,
        result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Offset,
    typename Result>
void offset(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Offset const& offset_,
    value_type<Result> const& fill_value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    offset<>(InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, offset_, fill_value, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern
