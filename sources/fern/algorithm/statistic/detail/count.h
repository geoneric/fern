#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/core/binary_aggregate_operation.h"
#include "fern/algorithm/statistic/sum.h"


namespace fern {
namespace algorithm {
namespace count {
namespace detail {

template<
    class Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)

    template<
        class Result>
    inline static void init(
        Value const& value1,
        Value const& value2,
        Result& result)
    {
        FERN_STATIC_ASSERT(std::is_integral, Result)
        FERN_STATIC_ASSERT(!std::is_same, Result, bool)

        result = value1 == value2 ? 1 : 0;
    }

    template<
        class Result>
    inline static void calculate(
        Value const& value1,
        Value const& value2,
        Result& result)
    {
        FERN_STATIC_ASSERT(std::is_integral, Result)
        FERN_STATIC_ASSERT(!std::is_same, Result, bool)

        result += value1 == value2 ? 1 : 0;
    }

};


struct Aggregator
{

    template<
        template<class, class, class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class ExecutionPolicy,
        class Value,
        class Result>
    inline static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        statistic::sum<OutOfRangePolicy>(input_no_data_policy,
            output_no_data_policy, execution_policy, value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void count(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    binary_aggregate_operation<Algorithm, Aggregator,
        binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            values, value, result);
}

} // namespace detail
} // namespace count
} // namespace algorithm
} // namespace fern
