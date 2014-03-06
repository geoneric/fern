#pragma once
#include <thread>
#include "fern/algorithm/core/operation_traits.h"


namespace fern {

class IndexRange
{

public:

    typedef size_t index;

    IndexRange()

        : _begin(),
          _end()

    {
    }

    IndexRange(
        index begin,
        index end)

        : _begin(begin),
          _end(end)

    {
    }

    index begin() const
    {
        return _begin;
    }

    index end() const
    {
        return _end;
    }

private:

    index          _begin;

    index          _end;

};


template<
    size_t nr_dimensions>
class IndexRanges
{

public:

    IndexRanges()

        : _ranges(nr_dimensions)

    {
    }

    IndexRanges(
        IndexRanges<nr_dimensions> const& other)
    {
        _ranges = other._ranges;
    }

    IndexRanges(
        IndexRange range1,
        IndexRange range2)

        : _ranges(nr_dimensions)

    {
        assert(nr_dimensions == 2);
        _ranges[0] = std::forward<IndexRange>(range1);
        _ranges[1] = std::forward<IndexRange>(range2);
    }

    IndexRanges<nr_dimensions>& operator=(
        IndexRanges<nr_dimensions>&& other)
    {
        if(&other != this) {
            _ranges = std::move(other._ranges);
        }
        return *this;
    }

    IndexRange const& operator[](
        size_t index) const
    {
        assert(index < _ranges.size());
        return _ranges[index];
    }

private:

    std::vector<IndexRange> _ranges;

};


std::vector<IndexRanges<2>> index_ranges(
    size_t const nr_threads,
    size_t const size1,
    size_t const size2)
{
    // Assuming the last size is the size of the dimension whose elements
    // are adjacent to each other in memory.
    size_t const block_width = size2;
    size_t const block_height = size1 / nr_threads;  // Integer division.
    size_t const nr_blocks = size1 / block_height;

    std::vector<IndexRanges<2>> ranges(nr_blocks);

    size_t row_offset = 0;
    for(size_t i = 0; i < nr_blocks; ++i) {
        ranges[i] = IndexRanges<2>(
            IndexRange(row_offset, row_offset + block_height),
            IndexRange(0, block_width));
        row_offset += block_height;
    }

    // Handle the remaining values.
    size_t const block_height_remainder = size1 - nr_blocks * block_height;

    if(block_height_remainder > 0) {
        ranges.emplace_back(
            IndexRange(row_offset, row_offset + block_height_remainder),
            IndexRange(0, block_width));
    }

    return std::move(ranges);
}


std::vector<IndexRanges<2>> index_ranges(
    size_t const size1,
    size_t const size2)
{
    size_t const nr_cores = std::thread::hardware_concurrency();
    size_t const nr_threads = nr_cores > 0 ? nr_cores : 2u;

    return std::move(index_ranges(nr_threads, size1, size2));
}


namespace serial {

template<
    class Operation,
    class A1,
    class A2,
    class R
>
void execute(
    Operation& operation,
    A1 const& argument1,
    A2 const& argument2,
    R& result)
{
    operation(argument1, argument2, result);
}

} // namespace serial


namespace concurrent {
namespace detail {
namespace dispatch {

// -----------------------------------------------------------------------------
template<
    class Operation,
    class A1,
    class A2,
    class R,
    class A1CollectionCategory,
    class A2CollectionCategory>
struct LocalOperationExecutor
{
};


template<
    class Operation,
    class A1,
    class A2,
    class R>
struct LocalOperationExecutor<
    Operation,
    A1,
    A2,
    R,
    array_2d_tag,
    constant_tag>
{

    inline static void execute(
        Operation& operation,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        // First argument can be blocked.
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(size1, size2);
        std::vector<std::thread> threads;

        for(auto const& block_range: ranges) {
            auto view_indices = std::make_tuple(
                Range(block_range[0].begin(), block_range[0].end()),
                Range(block_range[1].begin(), block_range[1].end()));
            threads.emplace_back(std::thread(std::ref(operation), view_indices,
                std::cref(argument1), std::cref(argument2), std::ref(result)));
        }

        for(auto& thread: threads) {
            thread.join();
        }
    }

};


template<
    class Operation,
    class A1,
    class A2,
    class R>
struct LocalOperationExecutor<
    Operation,
    A1,
    A2,
    R,
    array_2d_tag,
    array_2d_tag>
{

    inline static void execute(
        Operation& operation,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument1, 0) == fern::size(argument2, 0));
        assert(fern::size(argument1, 1) == fern::size(argument2, 1));

        // First argument and second argument can be blocked.
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(size1, size2);
        std::vector<std::thread> threads;

        for(auto const& block_range: ranges) {
            auto view_indices = std::make_tuple(
                Range(block_range[0].begin(), block_range[0].end()),
                Range(block_range[1].begin(), block_range[1].end()));
            threads.emplace_back(std::thread(std::ref(operation), view_indices,
                std::cref(argument1), std::cref(argument2), std::ref(result)));
        }

        for(auto& thread: threads) {
            thread.join();
        }
    }

};


// -----------------------------------------------------------------------------
template<
    class Operation,
    class A1,
    class A2,
    class R,
    class A1CollectionCategory,
    class A2CollectionCategory>
struct LocalAgregateOperationExecutor
{
};


template<
    class Operation,
    class A1,
    class A2,
    class R>
struct LocalAgregateOperationExecutor<
    Operation,
    A1,
    A2,
    R,
    array_2d_tag,
    constant_tag>
{

    inline static void execute(
        Operation& operation,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        // First argument can be blocked.
        // The result of each block must be stored in a collection. This
        // collection can be aggregated to the result. The operation determines
        // how this happens.
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(size1, size2);

        std::vector<std::thread> threads;
        std::vector<R> results_per_block(ranges.size(), 0);

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto view_indices = std::make_tuple(
                Range(block_range[0].begin(), block_range[0].end()),
                Range(block_range[1].begin(), block_range[1].end()));
            threads.emplace_back(std::thread(std::ref(operation), view_indices,
                std::cref(argument1), std::cref(argument2),
                std::ref(results_per_block[i])));
        }

        for(auto& thread: threads) {
            thread.join();
        }

        // Accumulate the results into one single result.
        Operation::aggregate(results_per_block, result);
    }

};


// -----------------------------------------------------------------------------
template<
    class Operation,
    class A1,
    class A2,
    class R,
    class OperationCategory>
struct Executor
{
};


template<
    class Operation,
    class A1,
    class A2,
    class R>
struct Executor<
    Operation,
    A1,
    A2,
    R,
    local_operation_tag>
{

    inline static void execute(
        Operation& operation,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        LocalOperationExecutor<Operation, A1, A2, R,
            typename ArgumentTraits<A1>::argument_category,
            typename ArgumentTraits<A2>::argument_category>::execute(operation,
                argument1, argument2, result);
    }

};


template<
    class Operation,
    class A1,
    class A2,
    class R>
struct Executor<
    Operation,
    A1,
    A2,
    R,
    local_aggregate_operation_tag>
{

    inline static void execute(
        Operation& operation,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        LocalAgregateOperationExecutor<Operation, A1, A2, R,
            typename ArgumentTraits<A1>::argument_category,
            typename ArgumentTraits<A2>::argument_category>::execute(operation,
                argument1, argument2, result);
    }

};

} // namespace dispatch
} // namespace detail


template<
    class Operation,
    class A1,
    class A2,
    class R
>
void execute(
    Operation& operation,
    A1 const& argument1,
    A2 const& argument2,
    R& result)
{
    detail::dispatch::Executor<Operation, A1, A2, R,
        typename OperationTraits<Operation>::category>::execute(operation,
            argument1, argument2, result);
}

} // namespace concurrent
} // namespace fern
