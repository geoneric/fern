#pragma once
#include "fern/core/base_class.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace algorithm {
namespace merge_no_data {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void merge_no_data_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& /* value */,
    Result& /* result */)
{
    if(!input_no_data_policy.is_no_data()) {
        if(input_no_data_policy.get<0>().is_no_data()) {
            output_no_data_policy.mark_as_no_data();
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void merge_no_data_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& /* value */,
    Result& /* result */)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!input_no_data_policy.is_no_data(i, j)) {
                if(input_no_data_policy.get<0>().is_no_data(i, j)) {
                    output_no_data_policy.mark_as_no_data(i, j);
                }
            }
        }
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
struct MergeNoDataByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
struct MergeNoDataByArgumentCategory<
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
        ExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        merge_no_data_0d(input_no_data_policy, output_no_data_policy,
            value, result);
    }

};


// TODO: 1d stuff.


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct MergeNoDataByArgumentCategory<
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
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        merge_no_data_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct MergeNoDataByArgumentCategory<
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
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                merge_no_data_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct MergeNoDataByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                MergeNoDataByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                SequentialExecutionPolicy>(execution_policy),
                            value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                MergeNoDataByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                ParallelExecutionPolicy>(execution_policy),
                            value, result);
                break;
            }
        }
    }

};

} // namespace dispatch


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void merge_no_data(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    dispatch::MergeNoDataByExecutionPolicy<InputNoDataPolicy,
        OutputNoDataPolicy, Value, Result>::
            apply(input_no_data_policy, output_no_data_policy,
                execution_policy, value, result);
}

} // namespace detail
} // namespace merge_no_data
} // namespace algorithm
} // namespace fern
