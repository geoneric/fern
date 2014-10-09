#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/binary_local_operation.h"


namespace fern {
namespace algorithm {
namespace greater {
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
        result = value1 > value2;
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
>
void greater(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    binary_local_operation<Algorithm,
        binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value1, value2, result);
}

} // namespace detail
} // namespace greater
} // namespace algorithm
} // namespace fern
