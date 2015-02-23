#pragma once
#include "fern/algorithm/core/unary_aggregate_operation.h"
#include "fern/algorithm/accumulator/min.h"


namespace fern {
namespace algorithm {
namespace unary_min {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void unary_min(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation<accumulator::Min,
        unary::DiscardDomainErrors, binary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace unary_min
} // namespace algorithm
} // namespace fern
