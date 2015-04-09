// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/core/binary_aggregate_operation.h"
#include "fern/algorithm/statistic/sum.h"


namespace fern {
namespace algorithm {
namespace count {
namespace detail {

template<
    typename Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)

    template<
        typename Result>
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
        typename Result>
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
        statistic::sum<OutOfRangePolicy>(input_no_data_policy,
            output_no_data_policy, execution_policy, value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void count(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
