// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <algorithm>
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/binary_local_operation.h"


namespace fern {
namespace algorithm {
namespace binary_max {
namespace detail {

template<
    typename Value1,
    typename Value2>
struct Algorithm
{

    template<
        typename Result>
    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        Result& result) const
    {
        result = std::max(value1, value2);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void binary_max(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    binary_local_operation<Algorithm,
        binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value1, value2, result);
}

} // namespace detail
} // namespace binary_max
} // namespace algorithm
} // namespace fern
