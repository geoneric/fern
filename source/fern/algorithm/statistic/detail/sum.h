#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/algorithm/core/unary_aggregate_operation.h"


namespace fern {
namespace algorithm {
namespace sum {
namespace detail {

template<
    typename Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(!std::is_same, Value, bool)

    template<
        typename Result>
    inline static void init(
        Value const& value,
        Result& result)
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = value;
    }

    template<
        typename Result>
    inline static void calculate(
        Value const& value,
        Result& result)
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result += value;
    }

};


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
    ExecutionPolicy const& execution_policy,
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
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        sum<OutOfRangePolicy>(input_no_data_policy, output_no_data_policy,
            execution_policy, value, result);
    }

};


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
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation<Algorithm, Aggregator,
        unary::DiscardDomainErrors, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace sum
} // namespace algorithm
} // namespace fern