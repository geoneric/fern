// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <tuple>
#include "fern/core/data_customization_point.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace unary_aggregate_operation_ {
namespace detail {

template<
    typename Accumulator,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename Value,
    bool out_of_range_risk>
class Adder
{

public:

    static bool    add                 (Accumulator& accumulator,
                                        Value const& value);

    static bool    merge               (Accumulator& accumulator,
                                        Accumulator&& other);

};


template<
    typename Accumulator,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename Value>
class Adder<
    Accumulator,
    OutOfRangePolicy,
    Value,
    false>
{

public:

    static inline bool add(
            Accumulator& accumulator,
            Value const& value)
    {
        accumulator(value);
        return true;
    }


    static inline bool merge(
        Accumulator& accumulator,
        Accumulator&& other)
    {
        accumulator |= other;
        return true;
    }

};


template<
    typename Accumulator,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename Value>
class Adder<
    Accumulator,
    OutOfRangePolicy,
    Value,
    true>
{

public:

    static inline bool add(
            Accumulator& accumulator,
            Value const& value)
    {
#if defined(_MSC_VER)
        return accumulator.operator()<OutOfRangePolicy>(value);
#else
        return accumulator.template operator()<OutOfRangePolicy>(value);
#endif
    }

    static inline bool merge(
        Accumulator& accumulator,
        Accumulator&& other)
    {
#if defined(_MSC_VER)
        return accumulator.operator|=<OutOfRangePolicy>(other);
#else
        return accumulator.template operator|=<OutOfRangePolicy>(other);
#endif
    }

};


template<
    typename Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename Value>
std::tuple<bool, bool, Accumulator> operation_0d(
    InputNoDataPolicy const& input_no_data_policy,
    Value const& value)
{
    using Adder_ = Adder<Accumulator, OutOfRangePolicy, value_type<Value>,
        Accumulator::out_of_range_risk>;

    bool result_within_range{true};
    bool accumulator_initialized{false};
    Accumulator accumulator;

    if(!std::get<0>(input_no_data_policy).is_no_data()) {
        result_within_range = Adder_::add(accumulator, get(value));
        accumulator_initialized = true;
    }

    return std::make_tuple(accumulator_initialized, result_within_range,
        std::move(accumulator));
}


template<
    typename Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename Value>
std::tuple<bool, bool, Accumulator> operation_1d(
    InputNoDataPolicy const& input_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value)
{
    using Adder_ = Adder<Accumulator, OutOfRangePolicy, value_type<Value>,
        Accumulator::out_of_range_risk>;

    bool accumulator_initialized{false};
    bool result_within_range{true};
    Accumulator accumulator;

    size_t const begin = index_ranges[0].begin();
    size_t const end = index_ranges[0].end();

    if(begin < end) {

        // Input is not empty, find first non-no-data value.
        for(size_t i = begin; i < end; ++i) {

            if(!std::get<0>(input_no_data_policy).is_no_data(i)) {

                result_within_range = Adder_::add(accumulator, get(value, i));
                accumulator_initialized = true;

                // Found first value, continue with the result.
                if(result_within_range) {

                    for(++i; i < end; ++i) {

                        if(!std::get<0>(input_no_data_policy).is_no_data(i)) {

                            result_within_range = Adder_::add(accumulator,
                                get(value, i));

                            if(!result_within_range) {
                                // No need to continue anymore.
                                break;
                            }
                        }
                    }
                }
            }

            if(!result_within_range) {
                break;
            }
        }
    }

    return std::make_tuple(accumulator_initialized, result_within_range,
        std::move(accumulator));
}


template<
    typename Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename Value>
std::tuple<bool, bool, Accumulator> operation_2d(
    InputNoDataPolicy const& input_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value)
{
    using Adder_ = Adder<Accumulator, OutOfRangePolicy, value_type<Value>,
        Accumulator::out_of_range_risk>;

    bool accumulator_initialized{false};
    bool result_within_range{true};
    Accumulator accumulator;

    size_t const begin1 = index_ranges[0].begin();
    size_t const end1 = index_ranges[0].end();
    size_t const begin2 = index_ranges[1].begin();
    size_t const end2 = index_ranges[1].end();

    if(begin1 < end1 && begin2 < end2) {

        // Input is not empty, find first non-no-data value.
        size_t i = begin1;
        size_t j = begin2;
        size_t index_;

        // Initialize result.
        for(; i < end1; ++i) {

            index_ = index(value, i, j);

            for(; j < end2; ++j) {

                if(!std::get<0>(input_no_data_policy).is_no_data(index_)) {

                    result_within_range = Adder_::add(accumulator,
                        get(value, index_));
                    accumulator_initialized = true;

                    // Found first value, continue with the result.
                    break;
                }

                ++index_;
            }

            if(accumulator_initialized) {
                break;
            }
        }

        // Continue where the previous loop stopped.
        if(accumulator_initialized) {
            ++j;

            for(; i < end1; ++i) {

                index_ = index(value, i, j);

                for(; j < end2; ++j) {

                    if(!std::get<0>(input_no_data_policy).is_no_data(index_)) {

                        result_within_range = Adder_::add(accumulator,
                            get(value, index_));

                        if(!result_within_range) {
                            // No need to continue anymore.
                            break;
                        }

                    }

                    ++index_;
                }

                if(j != end2) {
                    // This happens if the inner loop calls break.
                    // Set i and j such that all loops exit.
                    i = end1;
                    j = end2;
                }
                else {
                    j = begin2;
                }
            }
        }
    }

    return std::make_tuple(accumulator_initialized, result_within_range,
        std::move(accumulator));
}


namespace dispatch {

template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
struct UnaryAggregateOperation
{
};


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
struct UnaryAggregateOperation<
    Accumulator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        using Accumulator_ = Accumulator<value_type<Value>, value_type<Result>>;

        bool accumulator_initialized;
        bool result_within_range;
        Accumulator_ accumulator;

        std::tie(accumulator_initialized, result_within_range, accumulator) =
            operation_0d<Accumulator_, OutOfDomainPolicy, OutOfRangePolicy>(
                input_no_data_policy, value);

        if(accumulator_initialized && result_within_range) {
            get(result) = accumulator();
        }
        else {
            output_no_data_policy.mark_as_no_data();
        }
    }

};


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        using Accumulator_ = Accumulator<value_type<Value>, value_type<Result>>;

        bool result_within_range;
        bool accumulator_initialized;
        Accumulator_ accumulator;

        std::tie(accumulator_initialized, result_within_range, accumulator) =
            operation_1d<Accumulator_, OutOfDomainPolicy,
                OutOfRangePolicy>(input_no_data_policy, IndexRanges<1>{
                    IndexRange(0, size(value))}, value);

        if(accumulator_initialized && result_within_range) {
            get(result) = accumulator();
        }
        else {
            output_no_data_policy.mark_as_no_data();
        }
    }

};


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        using Accumulator_ = Accumulator<value_type<Value>, value_type<Result>>;
        using Adder_ = Adder<Accumulator_, OutOfRangePolicy, value_type<Value>,
            Accumulator_::out_of_range_risk>;

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<std::tuple<bool, bool, Accumulator_>>> futures;
        futures.reserve(ranges.size());

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_1d<Accumulator_, OutOfDomainPolicy, OutOfRangePolicy,
                    InputNoDataPolicy, Value>,
                std::cref(input_no_data_policy), std::cref(block_range),
                std::cref(value));
            futures.emplace_back(pool.submit(function));
        }

        std::vector<std::tuple<bool, bool, Accumulator_>> results;
        results.reserve(ranges.size());

        for(auto& future: futures) {
            results.emplace_back(future.get());
        }

        bool aggregated_result_within_range{false};
        bool aggregated_accumulator_initialized{false};
        Accumulator_ aggregated_accumulator;

        for(auto& result: results) {
            auto const& accumulator_initialized = get<0>(result);
            auto const& result_within_range = get<1>(result);

            if(accumulator_initialized && result_within_range) {
                auto& accumulator(get<2>(result));
                aggregated_accumulator_initialized = true;
                aggregated_result_within_range = Adder_::merge(
                    aggregated_accumulator, std::move(accumulator));
                if(!aggregated_result_within_range) {
                    break;
                }
            }
            else {
                // At least one block resulted in an out-of-range value.
                aggregated_result_within_range = false;
                break;
            }
        }

        if(aggregated_accumulator_initialized &&
                aggregated_result_within_range) {
            get(result) = aggregated_accumulator();
        }
        else {
            output_no_data_policy.mark_as_no_data();
        }
    }

};


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<SequentialExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
        }
    }

};


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        using Accumulator_ = Accumulator<value_type<Value>, value_type<Result>>;

        bool result_within_range;
        bool accumulator_initialized;
        Accumulator_ accumulator;

        std::tie(accumulator_initialized, result_within_range, accumulator) =
            operation_2d<Accumulator_, OutOfDomainPolicy,
                OutOfRangePolicy>(input_no_data_policy, IndexRanges<2>{
                    IndexRange(0, size(value, 0)),
                    IndexRange(0, size(value, 1))}, value);

        if(accumulator_initialized && result_within_range) {
            get(result) = accumulator();
        }
        else {
            output_no_data_policy.mark_as_no_data();
        }
    }

};


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        using Accumulator_ = Accumulator<value_type<Value>, value_type<Result>>;
        using Adder_ = Adder<Accumulator_, OutOfRangePolicy, value_type<Value>,
            Accumulator_::out_of_range_risk>;

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(value, 0);
        size_t const size2 = size(value, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<std::tuple<bool, bool, Accumulator_>>> futures;
        futures.reserve(ranges.size());

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_2d<Accumulator_, OutOfDomainPolicy, OutOfRangePolicy,
                    InputNoDataPolicy, Value>,
                std::cref(input_no_data_policy), std::cref(block_range),
                std::cref(value));
            futures.emplace_back(pool.submit(function));
        }

        std::vector<std::tuple<bool, bool, Accumulator_>> results;
        results.reserve(ranges.size());

        for(auto& future: futures) {
            results.emplace_back(future.get());
        }

        bool aggregated_result_within_range{false};
        bool aggregated_accumulator_initialized{false};
        Accumulator_ aggregated_accumulator;

        for(auto& result: results) {
            auto const& accumulator_initialized = get<0>(result);
            auto const& result_within_range = get<1>(result);

            if(accumulator_initialized && result_within_range) {
                auto& accumulator(get<2>(result));
                aggregated_accumulator_initialized = true;
                aggregated_result_within_range = Adder_::merge(
                    aggregated_accumulator, std::move(accumulator));
                if(!aggregated_result_within_range) {
                    break;
                }
            }
            else {
                // At least one block resulted in an out-of-range value.
                aggregated_result_within_range = false;
                break;
            }
        }

        if(aggregated_accumulator_initialized &&
                aggregated_result_within_range) {
            get(result) = aggregated_accumulator();
        }
        else {
            output_no_data_policy.mark_as_no_data();
        }
    }

};


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<SequentialExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
        }
    }

};

} // namespace dispatch
} // namespace detail
} // namespace unary_aggregate_operation_


/*!
    @ingroup    fern_algorithm_core_group
    @brief      Function that executes a unary aggregate operation.
    @tparam     Accumulator Class template of the operation to execute.
    @param[in]  value Input to pass to the operation.
    @param[out] result Output that is written by the operation.
    @sa         fern::algorithm::binary_aggregate_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    template<typename, typename> class Accumulator,
    template<typename> class OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void unary_aggregate_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation_::detail::dispatch::UnaryAggregateOperation<
        Accumulator,
        OutOfDomainPolicy<value_type<Value>>,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace algorithm
} // namespace fern
