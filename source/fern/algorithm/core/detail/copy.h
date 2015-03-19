#pragma once
#include "fern/core/base_class.h"
#include "fern/core/data_customization_point.h"
#include "fern/algorithm/core/data_traits/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


// Optimizations:
// - Copy of contiguous arrays with SkipNoData input no-data policy can call
//   std::copy/memcpy/...


namespace fern {
namespace algorithm {
namespace copy {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Source,
    typename Range,
    typename Destination,
    typename Position>
static void copy_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Source const& source,
    Range const& range,
    Destination& destination,
    Position const& position)
{
    value_type<Position> destination_index(get<0>(position));

    for(size_t i = range[0].begin(); i < range[0].end(); ++i) {

        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(destination_index);
        }
        else {
            get(destination, destination_index) = get(source, i);
        }

        ++destination_index;
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Source,
    typename Range,
    typename Destination,
    typename Position,
    typename ExecutionPolicy,
    typename SourceCollectionCategory>
struct CopyByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Source,
    typename Range,
    typename Destination,
    typename Position>
struct CopyByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Source,
    Range,
    Destination,
    Position,
    SequentialExecutionPolicy,
    array_1d_tag>
{

    // copy(1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Source const& source,
        Range const& range,
        Destination& destination,
        Position const& position)
    {
        assert(size(source) == size(destination));

        copy_1d(input_no_data_policy, output_no_data_policy,
            source, range, destination, position);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Source,
    typename Range,
    typename Destination,
    typename Position>
struct CopyByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Source,
    Range,
    Destination,
    Position,
    ParallelExecutionPolicy,
    array_1d_tag>
{

    // copy(1d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Source const& source,
        Range const& range,
        Destination& destination,
        Position const& position)
    {
        assert(size(source) == size(destination));

        // copy_1d(input_no_data_policy, output_no_data_policy,
        //     source, range, destination, position);

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(range);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        // The range passed in contains the indices of the elements to copy.
        // The index ranges just calculated, point to subsets of indices
        // of the range passed in. Per block we need to grab the indices to
        // copy from the range passed in, using the index ranges just
        // calculated.

        for(auto& block_range: ranges) {
            block_range = IndexRanges<1>{
                IndexRange(
                    range[0].begin() + block_range[0].begin(),
                    range[0].begin() + block_range[0].end())
            };
        }

        std::vector<Position> position_per_block(ranges.size());
        position_per_block[0] = position;

        for(size_t i = 1; i < ranges.size(); ++i) {
            position_per_block[i] = get<0>(position_per_block[i - 1]) +
                ranges[i - 1].size();
        }

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                copy_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Source, IndexRanges<1>, Destination, Position>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy),
                std::cref(source), std::cref(block_range),
                std::ref(destination), std::cref(position_per_block[i]));
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
    typename Source,
    typename Range,
    typename Destination,
    typename Position,
    typename ExecutionPolicy>
struct CopyByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Source const& source,
        Range const& range,
        Destination& destination,
        Position const& position)
    {
        CopyByArgumentCategory<
            InputNoDataPolicy,
            OutputNoDataPolicy,
            Source,
            Range,
            Destination,
            Position,
            ExecutionPolicy,
            base_class<argument_category<Source>, array_2d_tag>>
                ::apply(
                    input_no_data_policy, output_no_data_policy,
                    execution_policy, source, range, destination, position);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Source,
    typename Range,
    typename Destination,
    typename Position>
struct CopyByExecutionPolicy<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Source,
    Range,
    Destination,
    Position,
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Source const& source,
        Range const& range,
        Destination& destination,
        Position const& position)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                CopyByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Source,
                    Range,
                    Destination,
                    Position,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Source>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<SequentialExecutionPolicy>(
                                execution_policy),
                            source, range, destination, position);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                CopyByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Source,
                    Range,
                    Destination,
                    Position,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Source>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<ParallelExecutionPolicy>(
                                execution_policy),
                            source, range, destination, position);
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
    typename Source,
    typename Range,
    typename Destination,
    typename Position
>
void copy(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Source const& source,
    Range const& range,
    Destination& destination,
    Position const& position)
{
    dispatch::CopyByExecutionPolicy<InputNoDataPolicy, OutputNoDataPolicy,
        Source, Range, Destination, Position, ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            source, range, destination, position);
}

} // namespace detail
} // namespace copy
} // namespace algorithm
} // namespace fern
