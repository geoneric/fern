#pragma once
#include <algorithm>
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/binary_local_operation.h"


namespace fern {
namespace binary_min {
namespace detail {

template<
    class Value1,
    class Value2>
struct Algorithm
{

    template<
        class Result>
    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        Result& result) const
    {
        result = std::min(value1, value2);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void binary_min(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    binary_local_operation<Algorithm,
        binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value1, value2, result);

    // TODO
    /// n_ary_local_operation<Algorithm,
    ///     binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
    ///         input_no_data_policy, output_no_data_policy,
    ///         execution_policy,
    ///         result, std::forward<Values>(values)...);
}

} // namespace detail
} // namespace binary_min
} // namespace fern
