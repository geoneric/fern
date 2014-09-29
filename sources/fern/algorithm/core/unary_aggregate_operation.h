#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/thread_client.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace unary_aggregate_operation_ {
namespace detail {

template<
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Result& result)
{
    if(!input_no_data_policy.is_no_data()) {
        algorithm.init(fern::get(value), fern::get(result));
    }
    else {
        output_no_data_policy.mark_as_no_data();
    }
}


template<
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_1d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Result& result)
{
    using OORP = OutOfRangePolicy<value_type<Value>, value_type<Value>,
        value_type<Result>>;

    size_t const begin = index_ranges[0].begin();
    size_t const end = index_ranges[0].end();
    bool data_seen{false};

    if(begin < end) {

        value_type<Result> tmp_result;
        value_type<Result>& result_ = fern::get(result);

        for(size_t i = begin; i < end; ++i) {

            if(!input_no_data_policy.is_no_data(i)) {

                // Initialize result using the first valid value.
                algorithm.init(fern::get(value, i), tmp_result);
                result_ = tmp_result;
                data_seen = true;

                for(++i; i < end; ++i) {

                    if(!input_no_data_policy.is_no_data(i)) {

                        value_type<Value> const& a_value{fern::get(value, i)};
                        algorithm.calculate(a_value, tmp_result);

                        // lhs, rhs, lhs + rhs
                        if(!OORP::within_range(result_, a_value, tmp_result)) {
                            output_no_data_policy.mark_as_no_data();
                            break;
                        }

                        result_ = tmp_result;
                    }
                }
            }
        }
    }

    if(!data_seen) {
        output_no_data_policy.mark_as_no_data();
    }
}


template<
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    using OORP = OutOfRangePolicy<value_type<Value>, value_type<Value>,
        value_type<Result>>;

    size_t const begin1 = index_ranges[0].begin();
    size_t const end1 = index_ranges[0].end();
    size_t const begin2 = index_ranges[1].begin();
    size_t const end2 = index_ranges[1].end();
    bool data_seen{false};

    if(begin1 < end1 && begin2 < end2) {

        value_type<Result> tmp_result;
        value_type<Result>& result_ = fern::get(result);

        size_t i = begin1;
        size_t j = begin2;

        // Initialize result.
        for(; i < end1; ++i) {
            for(; j < end2; ++j) {

                if(!input_no_data_policy.is_no_data(i, j)) {

                    // Initialize result using the first valid value.
                    algorithm.init(fern::get(value, i, j), tmp_result);
                    result_ = tmp_result;
                    data_seen = true;
                    break;
                }
            }

            if(data_seen) {
                break;
            }
        }

        // Continue where the previous loop stopped.
        if(data_seen) {
            ++j;

            for(; i < end1; ++i) {
                for(; j < end2; ++j) {

                    if(!input_no_data_policy.is_no_data(i, j)) {

                        value_type<Value> const& a_value{fern::get(
                            value, i, j)};
                        algorithm.calculate(a_value, tmp_result);

                        // lhs, rhs, lhs + rhs
                        if(!OORP::within_range(result_, a_value,
                                tmp_result)) {
                            output_no_data_policy.mark_as_no_data();
                            break;
                        }

                        result_ = tmp_result;
                    }
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

    if(!data_seen) {
        output_no_data_policy.mark_as_no_data();
    }
}


namespace dispatch {

template<
    bool result_is_masking>
struct Aggregate
{
};


template<>
struct Aggregate<
    false>
{

    template<
        template<class, class, class> class OutOfRangePolicy,
        class Aggregator,
        class OutputNoDataPolicy,
        class ExecutionPolicy,
        class Results,
        class Result>
    static void apply(
        Aggregator const& aggregator,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Results const& results,
        Result& result)
    {
        Array<value_type<Result>, 1> results_(results);

        // Accumulate the results into one single result.
        // The final result is not masking, so the results aren't
        // either.
        aggregator.template apply<OutOfRangePolicy>(SkipNoData<>(),
            output_no_data_policy, sequential, results_, result);
    }

};


template<>
struct Aggregate<
    true>
{

    template<
        template<class, class, class> class OutOfRangePolicy,
        class Aggregator,
        class OutputNoDataPolicy,
        class ExecutionPolicy,
        class Results,
        class Result>
    static void apply(
        Aggregator const& aggregator,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Results const& results,
        Result& result)
    {
        MaskedArray<value_type<Result>, 1> results_(results);

        // Accumulate the results into one single result.
        // The final result is masking, so the results are also.
        aggregator.template apply<OutOfRangePolicy>(
            DetectNoDataByValue<Mask<1>>(results_.mask(), true),
            output_no_data_policy, sequential, results_, result);
    }

};


template<
    class Algorithm,
    class Aggregator,
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
struct UnaryAggregateOperation
{
};


template<
    class Algorithm,
    class Aggregator,
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
struct UnaryAggregateOperation<
    Algorithm,
    Aggregator,
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
        ExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            value, result);
    }

};


template<
    class Algorithm,
    class Aggregator,
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryAggregateOperation<
    Algorithm,
    Aggregator,
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
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value))},
            value, result);
    }

};


template<
    class Algorithm,
    class Aggregator,
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryAggregateOperation<
    Algorithm,
    Aggregator,
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
        ParallelExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Result> results_per_block(ranges.size(), Result(0));

        Algorithm algorithm;

        // TODO OutputNoDataPolicy must be created ad-hoc here. It must have
        //      an effect on the values in results_per_block.
        //      This depends on whether or not results are masking. Probably
        //      need to dispatch this whole function on that. First create a
        //      failing unit test that fails: include a block with only
        //      no-data and verify the result is not no-data.
        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_1d<OutOfDomainPolicy, OutOfRangePolicy, Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        Aggregator aggregator;

        Aggregate<ArgumentTraits<Result>::is_masking>::template
            apply<OutOfRangePolicy>(
                aggregator, output_no_data_policy, execution_policy,
                results_per_block, result);
    }

};


template<
    class Algorithm,
    class Aggregator,
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryAggregateOperation<
    Algorithm,
    Aggregator,
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
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(value, 0)),
                IndexRange(0, fern::size(value, 1))},
            value, result);
    }

};


template<
    class Algorithm,
    class Aggregator,
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryAggregateOperation<
    Algorithm,
    Aggregator,
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
        ParallelExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(value, 0);
        size_t const size2 = fern::size(value, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Result> results_per_block(ranges.size(), Result(0));

        Algorithm algorithm;

        // TODO OutputNoDataPolicy must be created ad-hoc here. It must have
        //      an effect on the values in results_per_block.
        //      This depends on whether or not results are masking. Probably
        //      need to dispatch this whole function on that. First create a
        //      failing unit test that fails: include a block with only
        //      no-data and verify the result is not no-data.
        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_2d<OutOfDomainPolicy, OutOfRangePolicy, Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        Aggregator aggregator;

        Aggregate<ArgumentTraits<Result>::is_masking>::template
            apply<OutOfRangePolicy>(
                aggregator, output_no_data_policy, execution_policy,
                results_per_block, result);
    }

};


template<
    class Algorithm,
    class Aggregator,
    class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryAggregateOperation<
    Algorithm,
    Aggregator,
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
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                UnaryAggregateOperation<
                    Algorithm,
                    Aggregator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            SequentialExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                UnaryAggregateOperation<
                    Algorithm,
                    Aggregator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag>::apply(
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
} // namespace detail
} // namespace unary_aggregate_operation_


//! Function that executes a unary aggregate operation.
/*!
    \tparam        Algorithm Class template of the operation to execute.
    \tparam        Aggregator Class of the aggregator.
    \param[in]     value Input to pass to the operation.
    \param[out]    result Output that is written by the operation.
    \sa            fern::binary_aggregate_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    template<class> class Algorithm,
    class Aggregator,
    template<class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void unary_aggregate_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation_::detail::dispatch::UnaryAggregateOperation<
        Algorithm<value_type<Value>>,
        Aggregator,
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
