#pragma once
#include <cmath>  // sin
#include "fern/core/assert.h"
#include "fern/algorithm/core/unary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace sin {
namespace detail {

template<
    typename Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

    template<
        typename Result>
    inline void operator()(
        Value const& value,
        Result& result) const
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = std::sin(value);
    }

};


template<
    template<typename> class OutOfDomainPolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void sin(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_local_operation<Algorithm,
        OutOfDomainPolicy, unary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value, result);
}

} // namespace detail
} // namespace sin
} // namespace algorithm
} // namespace fern
