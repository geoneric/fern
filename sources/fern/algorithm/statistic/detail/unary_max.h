#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/core/unary_aggregate_operation.h"


namespace fern {
namespace unary_max {
namespace detail {

template<
    class Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(!std::is_same, Value, bool)

    template<
        class Result>
    inline static void init(
        Value const& value,
        Result& result)
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = value;
    }

    template<
        class Result>
    inline static void calculate(
        Value const& value,
        Result& result)
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = std::max(result, value);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void unary_max(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result);


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
        unary_max<>(input_no_data_policy, output_no_data_policy,
            execution_policy, value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void unary_max(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation<Algorithm, Aggregator,
        unary::DiscardDomainErrors, binary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace unary_max
} // namespace fern
