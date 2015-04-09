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
#include "fern/algorithm/core/detail/copy.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Copy a @a range of elements from a @a source collection to a
                @a destination collection.

    The range of elements will be offset by @a position in destination.

    - The dimensionality of @a Source must be larger than 0.
    - @a Range must have the same dimensionality as @a Source.
    - The value type of @a Destination must be equal to the value type of
      @a Source.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Source,
    typename Range,
    typename Destination,
    typename Position>
void copy(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Source const& source,
    Range const& range,
    Destination& destination,
    Position const& position)
{
    FERN_STATIC_ASSERT(std::is_copy_assignable, value_type<Destination>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Destination>,
        value_type<Source>)
    static_assert(rank<Source>() > 0, "Rank must be larger than 0");
    static_assert(rank<Range>() == rank<Source>(),
       "Rank of range must equal rank of source");
    static_assert(rank<Destination>() == rank<Source>(),
        "Rank of range must equal rank of source");

    copy::detail::copy<>(input_no_data_policy,
        output_no_data_policy, execution_policy, source, range, destination,
        position);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Source,
    typename Range,
    typename Destination,
    typename Position>
void copy(
    ExecutionPolicy& execution_policy,
    Source const& source,
    Range const& range,
    Destination& destination,
    Position const& position)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    copy<>(InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        source, range, destination, position);
}

} // namespace core
} // namespace algorithm
} // namespace fern
