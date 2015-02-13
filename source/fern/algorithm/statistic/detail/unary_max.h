#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/core/unary_aggregate_operation.h"
#include "fern/algorithm/accumulator/max.h"


namespace fern {
namespace algorithm {
namespace unary_max {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void unary_max(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result);


struct Aggregator
{

    template<
        template<typename, typename, typename> class OutOfRangePolicy,
        typename InputNoDataPolicy,
        typename OutputNoDataPolicy,
        typename ExecutionPolicy,
        typename Value,
        typename Result>
    inline static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        unary_max<>(input_no_data_policy, output_no_data_policy,
            execution_policy, value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void unary_max(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation<accumulator::Max, Aggregator,
        unary::DiscardDomainErrors, binary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace unary_max
} // namespace algorithm
} // namespace fern
