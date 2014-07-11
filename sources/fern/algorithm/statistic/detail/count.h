#pragma once
#include <utility>
#include "fern/core/argument_categories.h"
#include "fern/core/argument_traits.h"
#include "fern/core/collection_traits.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace count {
namespace detail {

template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    if(!input_no_data_policy.is_no_data()) {
        fern::get(result) = fern::get(values) == fern::get(value) ? 1 : 0;
    }
    else {
        output_no_data_policy.mark_as_no_data();
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{

    size_t const begin = index_ranges[0].begin();
    size_t const end = index_ranges[0].end();
    bool data_seen{false};

    if(begin < end) {

        value_type<Result>& count = fern::get(result);
        count = 0;

        for(size_t i = begin; i < end; ++i) {

            if(!input_no_data_policy.is_no_data(i)) {

                count += fern::get(values, i) == value ? 1 : 0;
                data_seen = true;
            }
        }
    }

    if(!data_seen) {
        output_no_data_policy.mark_as_no_data();
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    size_t const begin1 = index_ranges[0].begin();
    size_t const end1 = index_ranges[0].end();
    size_t const begin2 = index_ranges[1].begin();
    size_t const end2 = index_ranges[1].end();
    bool data_seen{false};

    if(begin1 < end1 && begin2 < end2) {

        value_type<Result>& count = fern::get(result);
        count = 0;

        for(size_t i = begin1; i < end1; ++i) {
            for(size_t j = begin2; j < end2; ++j) {

                if(!input_no_data_policy.is_no_data(i, j)) {

                    count += fern::get(values, i, j) == value ? 1 : 0;
                    data_seen = true;
                }
            }
        }
    }

    if(!data_seen) {
        output_no_data_policy.mark_as_no_data();
    }
}


namespace dispatch {

template<
    class ExecutionPolicy,
    class ArrayCollectionCategory>
struct Count
{
};


template<
    class ExecutionPolicy>
struct Count<
    ExecutionPolicy,
    array_0d_tag>
{

    template<
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class Value,
        class Result>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        operation_0d<>(input_no_data_policy, output_no_data_policy,
            values, value, result);
    }

};


template<>
struct Count<
    SequentialExecutionPolicy,
    array_1d_tag>
{

    template<
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class Value,
        class Result>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        operation_1d<>(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(values))},
            values, value, result);
    }

};


template<>
struct Count<
    SequentialExecutionPolicy,
    array_2d_tag>
{

    template<
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class Value,
        class Result>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        operation_2d<>(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(values, 0)),
                IndexRange(0, fern::size(values, 1))},
            values, value, result);
    }

};


template<>
struct Count<
    ParallelExecutionPolicy,
    array_2d_tag>
{

    template<
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class Value,
        class Result>
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(values, 0);
        size_t const size2 = fern::size(values, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(values), std::cref(value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};

} // namespace dispatch


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
    // Dispatch on the collection category of values and the execution
    // policy.
    dispatch::Count<ExecutionPolicy, argument_category<Value>>::apply(
        input_no_data_policy, output_no_data_policy, execution_policy,
        values, value, result);
}

} // namespace detail
} // namespace count
} // namespace fern
