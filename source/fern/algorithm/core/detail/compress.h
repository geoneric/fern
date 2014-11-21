#pragma once
#include "fern/core/assert.h"
// #include "fern/core/array_2d_traits.h"
// #include "fern/core/constant_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/point.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/statistic/sum.h"
#include "fern/algorithm/core/copy.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


// Optimalisations:
// - Compression with SkipNoData input no-data policy is the same as copy.
//   Forward such a call. copy can be optimized too.


namespace fern {
namespace algorithm {
namespace compress {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename Value,
    typename Result,
    typename Count>
static void compress_1d(
    InputNoDataPolicy const& input_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Result& result,
    Count& count)
{
    size_t target_index{index_ranges[0].begin()};

    for(size_t source_index = index_ranges[0].begin();
            source_index < index_ranges[0].end(); ++source_index) {

        if(!input_no_data_policy.is_no_data(source_index)) {
            get(result, target_index) = get(value, source_index);
            ++target_index;
        }
    }

    count = target_index - index_ranges[0].begin();
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename Value,
    typename Result,
    typename Count,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
struct CompressByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename Value,
    typename Result,
    typename Count>
struct CompressByArgumentCategory<
    InputNoDataPolicy,
    Value,
    Result,
    Count,
    SequentialExecutionPolicy,
    array_1d_tag>
{

    // compress(1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result,
        Count& count)
    {
        compress_1d(input_no_data_policy,
            IndexRanges<1>{
                IndexRange(0, size(value)),
            }, value, result, count);
    }

};


template<
    typename InputNoDataPolicy,
    typename Value,
    typename Result,
    typename Count>
struct CompressByArgumentCategory<
    InputNoDataPolicy,
    Value,
    Result,
    Count,
    ParallelExecutionPolicy,
    array_1d_tag>
{

    // compress(1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result,
        Count& count)
    {
        assert(size(value) <= size(result));

        // Compress each region individually.
        ThreadPool& pool(ThreadClient::pool());
        size_t const size_ = size(result);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Count> counts_per_block(ranges.size());

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                compress_1d<InputNoDataPolicy, Value, Result, Count>,
                std::cref(input_no_data_policy),
                std::cref(block_range), std::cref(value), std::ref(result),
                std::ref(counts_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        // Copy all results per region to the front of the final result.
        Point<Count, 1> position{0};

        for(size_t i = 1; i < ranges.size(); ++i) {
            // Define range of values to copy. block_range is almost it, but
            // we need to use the number of compressed values instead of the
            // full range.
            assert(counts_per_block[i] <= ranges[i].size());
            auto& block_range(ranges[i]);
            block_range[0].set_end(block_range[0].begin() +
                counts_per_block[i]);

            // Define position to copy to.
            get<0>(position) += counts_per_block[i-1];
            assert(block_range[0].begin() >= get<0>(position));

            // Copy this block's compress result to the front of the overall
            // result collection.
            core::copy(sequential, result, block_range, result, position);
        }

        statistic::sum(sequential, counts_per_block, count);
    }

};


// template<
//     typename InputNoDataPolicy,
//     typename OutputNoDataPolicy,
//     typename Value1,
//     typename Value2,
//     typename Result>
// struct CompressByArgumentCategory<
//     InputNoDataPolicy,
//     OutputNoDataPolicy,
//     Value1,
//     Value2,
//     Result,
//     ParallelExecutionPolicy,
//     array_2d_tag,
//     array_0d_tag>
// {
// 
//     // compress(2d, 0d)
//     static void apply(
//         InputNoDataPolicy const& input_no_data_policy,
//         OutputNoDataPolicy& output_no_data_policy,
//         ParallelExecutionPolicy const& /* execution_policy */,
//         Value1 const& value1,
//         Value2 const& value2,
//         Result& result)
//     {
//         assert(size(value1, 0) == size(result, 0));
//         assert(size(value1, 1) == size(result, 1));
// 
//         ThreadPool& pool(ThreadClient::pool());
//         size_t const size1 = size(result, 0);
//         size_t const size2 = size(result, 1);
//         std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
//             size1, size2);
//         std::vector<std::future<void>> futures;
//         futures.reserve(ranges.size());
// 
//         for(auto const& block_range: ranges) {
//             auto function = std::bind(
//                 cover_2d_0d<InputNoDataPolicy, OutputNoDataPolicy,
//                     Value1, Value2, Result>,
//                 std::cref(input_no_data_policy),
//                 std::ref(output_no_data_policy), std::cref(block_range),
//                 std::cref(value1), std::cref(value2), std::ref(result));
//             futures.emplace_back(pool.submit(function));
//         }
// 
//         for(auto& future: futures) {
//             future.get();
//         }
//     }
// 
// };
// 
// 
// template<
//     typename InputNoDataPolicy,
//     typename OutputNoDataPolicy,
//     typename Value1,
//     typename Value2,
//     typename Result>
// struct CompressByArgumentCategory<
//     InputNoDataPolicy,
//     OutputNoDataPolicy,
//     Value1,
//     Value2,
//     Result,
//     SequentialExecutionPolicy,
//     array_2d_tag,
//     array_2d_tag>
// {
// 
//     // compress(2d, 2d)
//     static void apply(
//         InputNoDataPolicy const& input_no_data_policy,
//         OutputNoDataPolicy& output_no_data_policy,
//         SequentialExecutionPolicy const& /* execution_policy */,
//         Value1 const& value1,
//         Value2 const& value2,
//         Result& result)
//     {
//         assert(size(value1, 0) == size(result, 0));
//         assert(size(value1, 1) == size(result, 1));
//         assert(size(value2, 0) == size(result, 0));
//         assert(size(value2, 1) == size(result, 1));
// 
//         cover_2d_2d(input_no_data_policy, output_no_data_policy,
//             IndexRanges<2>{
//                 IndexRange(0, size(result, 0)),
//                 IndexRange(0, size(result, 1)),
//             }, value1, value2, result);
//     }
// 
// };
// 
// 
// template<
//     typename InputNoDataPolicy,
//     typename OutputNoDataPolicy,
//     typename Value1,
//     typename Value2,
//     typename Result>
// struct CompressByArgumentCategory<
//     InputNoDataPolicy,
//     OutputNoDataPolicy,
//     Value1,
//     Value2,
//     Result,
//     ParallelExecutionPolicy,
//     array_2d_tag,
//     array_2d_tag>
// {
// 
//     // compress(2d, 2d)
//     static void apply(
//         InputNoDataPolicy const& input_no_data_policy,
//         OutputNoDataPolicy& output_no_data_policy,
//         ParallelExecutionPolicy const& /* execution_policy */,
//         Value1 const& value1,
//         Value2 const& value2,
//         Result& result)
//     {
//         assert(size(value1, 0) == size(result, 0));
//         assert(size(value1, 1) == size(result, 1));
//         assert(size(value2, 0) == size(result, 0));
//         assert(size(value2, 1) == size(result, 1));
// 
//         ThreadPool& pool(ThreadClient::pool());
//         size_t const size1 = size(result, 0);
//         size_t const size2 = size(result, 1);
//         std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
//             size1, size2);
//         std::vector<std::future<void>> futures;
//         futures.reserve(ranges.size());
// 
//         for(auto const& block_range: ranges) {
//             auto function = std::bind(
//                 cover_2d_2d<InputNoDataPolicy, OutputNoDataPolicy,
//                     Value1, Value2, Result>,
//                 std::cref(input_no_data_policy),
//                 std::ref(output_no_data_policy), std::cref(block_range),
//                 std::cref(value1), std::cref(value2), std::ref(result));
//             futures.emplace_back(pool.submit(function));
//         }
// 
//         for(auto& future: futures) {
//             future.get();
//         }
//     }
// 
// };


template<
    typename InputNoDataPolicy,
    typename Value,
    typename Result,
    typename Count>
struct CompressByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result,
        Count& count)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                CompressByArgumentCategory<
                    InputNoDataPolicy,
                    Value,
                    Result,
                    Count,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                SequentialExecutionPolicy>(execution_policy),
                            value, result, count);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                CompressByArgumentCategory<
                    InputNoDataPolicy,
                    Value,
                    Result,
                    Count,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                ParallelExecutionPolicy>(execution_policy),
                            value, result, count);
                break;
            }
        }
    }

};

} // namespace dispatch


template<
    typename InputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result,
    typename Count>
static void compress(
    InputNoDataPolicy const& input_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result,
    Count& count)
{
    dispatch::CompressByExecutionPolicy<InputNoDataPolicy,
        Value, Result, Count>::apply(
            input_no_data_policy, execution_policy, value, result, count);
}

} // namespace detail
} // namespace compress
} // namespace algorithm
} // namespace fern
