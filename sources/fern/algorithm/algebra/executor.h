#pragma once
#include <thread>
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/core/operation_traits.h"


namespace fern {
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


// operation(array_2d, constant)
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
        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(std::ref(operation), block_range,
                std::cref(argument1), std::cref(argument2), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


// operation(array_2d, array_2d)
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
        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(std::ref(operation), block_range,
                std::cref(argument1), std::cref(argument2), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
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


// operation(array_2d, constant)
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
        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<R> results_per_block(ranges.size(), 0);

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(std::ref(operation), block_range,
                std::cref(argument1), std::cref(argument2),
                std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
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
