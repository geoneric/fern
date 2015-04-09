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
#include "fern/algorithm/core/fill.h"
#include "fern/algorithm/core/nullary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace defined {
namespace detail {

struct Algorithm
{

    template<
        typename Result>
    inline void operator()(
        Result& result) const
    {
        result = Result(1);
    }

};


template<
    typename InputNoDataPolicy,
    typename ExecutionPolicy,
    typename Result
>
void defined(
    InputNoDataPolicy const& input_no_data_policy,
    ExecutionPolicy& execution_policy,
    Result& result)
{
    // 1. Whatever the input-no-data policy, fill result with false.
    // 2. Let algorithm return true for each non-no-data value.

    fern::algorithm::core::fill(execution_policy, value_type<Result>(0),
        result);

    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    nullary_local_operation<Algorithm>(
        input_no_data_policy, output_no_data_policy, execution_policy, result);
}

} // namespace detail
} // namespace defined
} // namespace algorithm
} // namespace fern
