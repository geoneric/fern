#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/core/fill.h"
#include "fern/algorithm/core/nullary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace defined {
namespace detail {

struct Algorithm
{

    template<
        class Result>
    inline void operator()(
        Result& result) const
    {
        result = Result(1);
    }

};


template<
    class InputNoDataPolicy,
    class ExecutionPolicy,
    class Result
>
void defined(
    InputNoDataPolicy const& input_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Result& result)
{
    // 1. Whatever the input-no-data policy, fill result with false.
    // 2. Let algorithm return true for each non-no-data value.

    fern::core::fill(execution_policy, Result(0), result);

    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    nullary_local_operation<Algorithm>(
        input_no_data_policy, output_no_data_policy, execution_policy, result);
}

} // namespace detail
} // namespace defined
} // namespace fern
