#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/unary_disaggregate_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace fill {
namespace detail {

template<
    typename Value>
struct Algorithm
{

    template<
        typename Result>
    inline void operator()(
        Value const& value,
        Result& result) const
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = value;
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void fill(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_disaggregate_operation<Algorithm,
        unary::DiscardDomainErrors, unary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value, result);
}

} // namespace detail
} // namespace fill
} // namespace algorithm
} // namespace fern
