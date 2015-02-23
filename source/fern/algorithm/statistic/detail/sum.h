#pragma once
#include "fern/algorithm/core/unary_aggregate_operation.h"
#include "fern/algorithm/accumulator/sum.h"


namespace fern {
namespace algorithm {
namespace sum {
namespace detail {

template<
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void sum(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation<accumulator::Sum,
        unary::DiscardDomainErrors, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace sum
} // namespace algorithm
} // namespace fern
