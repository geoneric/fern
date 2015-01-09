#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/point.h"
#include "fern/core/thread_client.h"
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

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        if(!std::get<0>(input_no_data_policy).is_no_data(i)) {
            get(result, target_index) = get(value, i);
            ++target_index;
        }
    }

    count = target_index - index_ranges[0].begin();
}


template<
    typename InputNoDataPolicy,
    typename Value,
    typename Result,
    typename Count>
static void compress_2d(
    InputNoDataPolicy const& input_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result,
    Count& count)
{
    // i = row * nr_cols + col
    size_t const initial_index{index_ranges[0].begin() * size(value, 1) +
        index_ranges[1].begin()};
    size_t target_index{initial_index};

    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(value, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!std::get<0>(input_no_data_policy).is_no_data(index_)) {
                get(result, target_index) = get(value, index_);
                ++target_index;
            }

            ++index_;
        }
    }

    count = target_index - initial_index;
}


// Copy all results per region to the front of the final result.
template<
    typename Result,
    typename Count,
    size_t nr_dimensions>
void copy_regional_results_to_front(
    std::vector<IndexRanges<nr_dimensions>> const& ranges,
    std::vector<Count> const& counts_per_block,
    Result& result,
    Count& count)
{
    // Start position of elements to copy from.
    size_t source_index{0};

    // Start position of elements to copy to.
    size_t target_index{0};

    for(size_t i = 0; i < ranges.size(); ++i) {
        // Range of indices in current region.
        auto const& block_range(ranges[i]);

        // Number of values compressed must be <= number of values in
        // the region.
        assert(counts_per_block[i] <= block_range.size());

        // Range of indices of compressed values in 1D result.
        IndexRanges<1> compressed_values_range{IndexRange(source_index,
            source_index + block_range.size())};

        core::copy(sequential, result, compressed_values_range, result,
            Point<size_t, 1>{target_index});

        source_index += block_range.size();
        target_index += counts_per_block[i];
    }

    count = target_index;
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

        copy_regional_results_to_front(ranges, counts_per_block, result, count);
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
    SequentialExecutionPolicy,
    array_2d_tag>
{

    // compress(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result,
        Count& count)
    {
        compress_2d(input_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value, 0)),
                IndexRange(0, size(value, 1)),
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
    array_2d_tag>
{

    // compress(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result,
        Count& count)
    {
        // Compress each region individually.
        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(value, 0);
        size_t const size2 = size(value, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Count> counts_per_block(ranges.size());

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                compress_2d<InputNoDataPolicy, Value, Result, Count>,
                std::cref(input_no_data_policy),
                std::cref(block_range), std::cref(value), std::ref(result),
                std::ref(counts_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        copy_regional_results_to_front(ranges, counts_per_block, result, count);
    }

};


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
