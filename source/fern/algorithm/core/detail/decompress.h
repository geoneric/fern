#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace decompress {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
static void decompress_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Result& result)
{
    size_t source_index{index_ranges[0].begin()};

    for(size_t target_index = index_ranges[0].begin();
            target_index < index_ranges[0].end(); ++target_index) {

        if(input_no_data_policy.is_no_data(target_index)) {
            output_no_data_policy.mark_as_no_data(target_index);
        }
        else {
            get(result, target_index) = get(value, source_index);
            ++source_index;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
static void decompress_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    // i = row * nr_cols + col
    size_t source_index{index_ranges[0].begin() * size(result, 1) +
        index_ranges[1].begin()};

    for(size_t target_index1 = index_ranges[0].begin();
            target_index1 < index_ranges[0].end(); ++target_index1) {
        for(size_t target_index2 = index_ranges[1].begin();
                target_index2 < index_ranges[1].end(); ++target_index2) {

            if(input_no_data_policy.is_no_data(target_index1, target_index2)) {
                output_no_data_policy.mark_as_no_data(target_index1,
                    target_index2);
            }
            else {
                get(result, target_index1, target_index2) = get(value,
                    source_index);
                ++source_index;
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
    typename ResultCollectionCategory>
struct DeCompressByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct DeCompressByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag>
{

    // decompress(1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        decompress_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{
                IndexRange(0, size(result)),
            }, value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct DeCompressByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag>
{

    // decompress(1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        // TODO Parallelize?
        decompress_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{
                IndexRange(0, size(result)),
            }, value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct DeCompressByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag>
{

    // decompress(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        decompress_2d(input_no_data_policy, output_no_data_policy,
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
struct DeCompressByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag>
{

    // decompress(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        // TODO Parallelize?
        decompress_2d(input_no_data_policy, output_no_data_policy,
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
struct DeCompressByExecutionPolicy
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
                DeCompressByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Result>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                SequentialExecutionPolicy>(execution_policy),
                            value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                DeCompressByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Result>, array_2d_tag>>
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
static void decompress(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    dispatch::DeCompressByExecutionPolicy<InputNoDataPolicy,
        OutputNoDataPolicy, Value, Result>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace decompress
} // namespace algorithm
} // namespace fern
