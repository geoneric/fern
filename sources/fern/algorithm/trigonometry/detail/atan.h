#pragma once
#include <cmath>  // atan
#include "fern/core/assert.h"
#include "fern/algorithm/core/unary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace atan {
namespace detail {

template<
    class Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

    template<
        class Result>
    inline void operator()(
        Value const& value,
        Result& result) const
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = std::atan(value);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void atan(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_local_operation<Algorithm,
        unary::DiscardDomainErrors, unary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value, result);
}

} // namespace detail
} // namespace atan
} // namespace fern
