#pragma once
#include <thread>
#include <boost/mpl/if.hpp>
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/detect_no_data_by_value.h"
#include "fern/algorithm/policy/skip_no_data.h"


namespace fern {
namespace serial {

template<
    class Operation,
    class A,
    class R
>
void execute(
    Operation& operation,
    A const& argument,
    R& result)
{
    operation(argument, result);
}


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

template<
    class Operation,
    class R,
    bool masking_result>
struct Aggregate
{
};


template<
    class Operation,
    class R>
struct Aggregate<
    Operation,
    R,
    true>
{

    template<
        class Collection>
    inline static void aggregate(
        Operation& operation,
        Collection const& results_per_block,
        R& result)
    {
        using MaskedArray = MaskedArray<
            typename ArgumentTraits<R>::value_type, 1>;
        MaskedArray results(results_per_block);

        // Accumulate the results into one single result.
        // The result is masking, to the results_per_block are also.
        operation.aggregate(DetectNoDataByValue<Mask<1>>(results.mask(), true),
            results, result);
    }

};


template<
    class Operation,
    class R>
struct Aggregate<
    Operation,
    R,
    false>
{

    template<
        class Collection>
    inline static void aggregate(
        Operation& operation,
        Collection const& results_per_block,
        R& result)
    {
        Array<typename ArgumentTraits<R>::value_type, 1> results(
            results_per_block);

        // Accumulate the results into one single result.
        // The final result is not masking, so the results_per_block aren't
        // either.
        operation.aggregate(SkipNoData(), results, result);
    }

};


namespace dispatch {

// -----------------------------------------------------------------------------
template<
    class Operation,
    class A,
    class R,
    class ACollectionCategory>
struct UnaryLocalOperationExecutor
{
};


// -----------------------------------------------------------------------------
template<
    class Operation,
    class A1,
    class A2,
    class R,
    class A1CollectionCategory,
    class A2CollectionCategory>
struct BinaryLocalOperationExecutor
{
};


// operation(array_2d, constant)
template<
    class Operation,
    class A1,
    class A2,
    class R>
struct BinaryLocalOperationExecutor<
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
struct BinaryLocalOperationExecutor<
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
    class A,
    class R,
    class ACollectionCategory>
struct UnaryLocalAggregateOperationExecutor
{
};


/// // operation(constant)
/// template<
///     class Operation,
///     class A,
///     class R>
/// struct UnaryLocalAggregateOperationExecutor<
///     Operation,
///     A,
///     R,
///     constant_tag>
/// {
/// 
///     inline static void execute(
///         Operation& operation,
///         A const& argument,
///         R& result)
///     {
///         Operation(argument, result);
///     }
/// 
/// };


// operation(array_2d)
template<
    class Operation,
    class A,
    class R>
struct UnaryLocalAggregateOperationExecutor<
    Operation,
    A,
    R,
    array_2d_tag>
{

    inline static void execute(
        Operation& operation,
        A const& argument,
        R& result)
    {
        // Argument can be blocked.
        // The result of each block must be stored in a collection. This
        // collection can be aggregated to the result. The operation determines
        // how this happens.
        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(argument, 0);
        size_t const size2 = fern::size(argument, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<R> results_per_block(ranges.size(), R(0));

        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(std::ref(operation), block_range,
                std::cref(argument), std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        // TODO When aggregating to a single value, execution can stop when
        //      one thread results in a no-data result.
        for(auto& future: futures) {
            future.get();
        }

        // Dispatch on mask-ness of the result type. If masked, then we need
        // to create a masked array and configure an InputMaskByValue policy.
        // Otherwise, things get simpler, but still different. We need to
        // dispatch.
        // TODO Rename aggregate to reduce? Check map-reduce literature.
        Aggregate<Operation, R, ArgumentTraits<R>::is_masking>::aggregate(
            operation, results_per_block, result);
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
struct BinaryLocalAggregateOperationExecutor
{
};


// operation(array_2d, constant)
template<
    class Operation,
    class A1,
    class A2,
    class R>
struct BinaryLocalAggregateOperationExecutor<
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
        std::vector<R> results_per_block(ranges.size(), R(0));

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

        /// // Accumulate the results into one single result.
        /// Operation::aggregate(results_per_block, result);

        // Dispatch on mask-ness of the result type. If masked, then we need
        // to create a masked array and configure an InputMaskByValue policy.
        // Otherwise, things get simpler, but still different. We need to
        // dispatch.
        // TODO Rename aggregate to reduce? Check map-reduce literature.
        Aggregate<Operation, R, ArgumentTraits<R>::is_masking>
            ::aggregate(operation, results_per_block, result);
    }

};


// -----------------------------------------------------------------------------
template<
    class Operation,
    class A,
    class R,
    class OperationCategory>
struct UnaryExecutor
{
};


template<
    class Operation,
    class A,
    class R>
struct UnaryExecutor<
    Operation,
    A,
    R,
    local_operation_tag>
{

    inline static void execute(
        Operation& operation,
        A const& argument,
        R& result)
    {
        UnaryLocalOperationExecutor<Operation, A, R,
            typename ArgumentTraits<A>::argument_category>::execute(operation,
                argument, result);
    }

};


template<
    class Operation,
    class A,
    class R>
struct UnaryExecutor<
    Operation,
    A,
    R,
    local_aggregate_operation_tag>
{

    inline static void execute(
        Operation& operation,
        A const& argument,
        R& result)
    {
        UnaryLocalAggregateOperationExecutor<Operation, A, R,
            typename ArgumentTraits<A>::argument_category>::execute(operation,
                argument, result);
    }

};


// -----------------------------------------------------------------------------
template<
    class Operation,
    class A1,
    class A2,
    class R,
    class OperationCategory>
struct BinaryExecutor
{
};


template<
    class Operation,
    class A1,
    class A2,
    class R>
struct BinaryExecutor<
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
        BinaryLocalOperationExecutor<Operation, A1, A2, R,
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
struct BinaryExecutor<
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
        BinaryLocalAggregateOperationExecutor<Operation, A1, A2, R,
            typename ArgumentTraits<A1>::argument_category,
            typename ArgumentTraits<A2>::argument_category>::execute(operation,
                argument1, argument2, result);
    }

};

} // namespace dispatch
} // namespace detail


template<
    class Operation,
    class A,
    class R
>
void execute(
    Operation& operation,
    A const& argument,
    R& result)
{
    detail::dispatch::UnaryExecutor<Operation, A, R,
        typename OperationTraits<Operation>::category>::execute(operation,
            argument, result);
}


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
    detail::dispatch::BinaryExecutor<Operation, A1, A2, R,
        typename OperationTraits<Operation>::category>::execute(operation,
            argument1, argument2, result);
}

} // namespace concurrent
} // namespace fern
